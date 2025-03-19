import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class DyT(nn.Module):
	def __init__(self, C, init_α):
		super().__init__()
		self.alpha = nn.Parameter(ones(1) * init_alpha)
		self.gamma = nn.Parameter(ones(C))
		self.beta = nn.Parameter(zeros(C))

	def forward(self, x):
		x = tanh(self.alpha * x)

		return self.gamma * x + self.beta

class FeedForward(nn.Module):
	def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
		super().__init__()

		# default scaling down by 2/3 since normal 
		# d_ff is 4xd_model
		# Should be ~2.667 scalling now
		# based on Llama SwiGLU FeedForward
		# https://github.com/meta-llama/llama
		d_ff = int(2 * d_ff // 3)

		# TODO: checking out parallel Linear from llama
		self.fc_in = nn.Linear(d_model, d_ff * 2) # can be column parallel
		self.fc_out = nn.Linear(d_ff, d_model) # can be row parallel

		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_proj = self.fc_in(x)
		gate, x_proj = x_proj.chunk(2, dim=-1)

		x = x_proj * F.silu(gate)
		x = self.fc_out(self.dropout(x))

		return x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device)  # type: ignore
	freqs = torch.outer(t, freqs).float()  # type: ignore
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	ndim = x.ndim
	assert 0 <= 1 < ndim
	assert freqs_cis.shape == (x.shape[1], x.shape[-1])

	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

	return freqs_cis.view(*shape)

def apply_rotary_emb(
	xq: torch.Tensor,
	xk: torch.Tensor,
	freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
	freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

	return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
		model_parallel_size = fs_init.get_model_parallel_world_size()
		self.n_local_heads = args.n_heads // model_parallel_size
		self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
		self.n_rep = self.n_local_heads // self.n_local_kv_heads
		self.head_dim = args.dim // args.n_heads

		self.wq = ColumnParallelLinear(
			args.dim,
			args.n_heads * self.head_dim,
			bias=False,
			gather_output=False,
			init_method=lambda x: x,
		)
		self.wk = ColumnParallelLinear(
			args.dim,
			self.n_kv_heads * self.head_dim,
			bias=False,
			gather_output=False,
			init_method=lambda x: x,
		)
		self.wv = ColumnParallelLinear(
			args.dim,
			self.n_kv_heads * self.head_dim,
			bias=False,
			gather_output=False,
			init_method=lambda x: x,
		)
		self.wo = RowParallelLinear(
			args.n_heads * self.head_dim,
			args.dim,
			bias=False,
			input_is_parallel=True,
			init_method=lambda x: x,
		)

		self.cache_k = torch.zeros(
			(
				args.max_batch_size,
				args.max_seq_len,
				self.n_local_kv_heads,
				self.head_dim,
			)
		).cuda()
		self.cache_v = torch.zeros(
			(
				args.max_batch_size,
				args.max_seq_len,
				self.n_local_kv_heads,
				self.head_dim,
			)
		).cuda()

	def forward(
		self,
		x: torch.Tensor,
		start_pos: int,
		freqs_cis: torch.Tensor,
		mask: Optional[torch.Tensor],
	):
		bsz, seqlen, _ = x.shape
		xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

		xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
		xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
		xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

		xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

		self.cache_k = self.cache_k.to(xq)
		self.cache_v = self.cache_v.to(xq)

		self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
		self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

		keys = self.cache_k[:bsz, : start_pos + seqlen]
		values = self.cache_v[:bsz, : start_pos + seqlen]

		# repeat k/v heads if n_kv_heads < n_heads
		keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
		values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

		xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
		keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
		values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
		scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
		if mask is not None:
			scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
		scores = F.softmax(scores.float(), dim=-1).type_as(xq)
		output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
		return self.wo(output)

