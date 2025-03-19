import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Tuple

class DyT(nn.Module):
	def __init__(self, C, init_alpha=0.5):
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
		self.gamma = nn.Parameter(torch.ones(C))
		self.beta = nn.Parameter(torch.zeros(C))

	def forward(self, x):
		x = torch.tanh(self.alpha * x)

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

	assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
		f"freqs_cis shape {freqs_cis.shape} needs to be {(x.shape[1], x.shape[-1])}"
	)

	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

	return freqs_cis.view(*shape)

def apply_rotary_emb(
	xq: torch.Tensor,
	xk: torch.Tensor,
	freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	xq_complex = torch.view_as_complex(
		rearrange(xq.float(), "... (n two) -> ... n two", two=2)
	)
	xk_complex = torch.view_as_complex(
		rearrange(xk.float(), "... (n two) -> ... n two", two=2)
	)

	freqs_cis_q = reshape_for_broadcast(freqs_cis[:xq.shape[1]], xq_complex).to(xq.device)
	freqs_cis_k = reshape_for_broadcast(freqs_cis[:xk.shape[1]], xk_complex).to(xk.device)

	xq_out = torch.view_as_real(xq_complex * freqs_cis_q).flatten(3)
	xk_out = torch.view_as_real(xk_complex * freqs_cis_k).flatten(3)

	return xq_out.type_as(xq), xk_out.type_as(xk)
