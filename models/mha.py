import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import xformers.ops as xops

from models.positional_encoding import apply_rotary_emb

# TODO: Implement MHA alternative with Convexifying attention and RWKV version to compare it with default one
class KVCacheMHA(nn.Module):
	def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 256, max_batch_size: int = 256):
		super().__init__()
		if d_model % n_heads != 0:
			raise ValueError("d_model must be divisible by n_heads")

		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads
		self.scaling = self.head_dim ** -0.5
		self.max_seq_len = max_seq_len
		self.max_batch_size = max_batch_size

		self.q_proj = nn.Linear(d_model, d_model)
		self.k_proj = nn.Linear(d_model, d_model)
		self.v_proj = nn.Linear(d_model, d_model)
		self.out_proj = nn.Linear(d_model, d_model)
		self.p = dropout

		cache_dtype = next(self.parameters()).dtype

		self.register_buffer(
			"cache_k",
			torch.zeros(max_batch_size, n_heads, max_seq_len, self.head_dim, dtype=cache_dtype),
			persistent=False,
		)
		self.register_buffer(
			"cache_v",
			torch.zeros(max_batch_size, n_heads, max_seq_len, self.head_dim, dtype=cache_dtype),
			persistent=False,
		)

	def forward(
		self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
		attn_mask: torch.Tensor = None, is_causal: bool = False,
		kv_write_indices: torch.Tensor = None,
		start_pos: int = 0,
		freqs_cis: torch.Tensor = None,
		**kwargs,
	) -> torch.Tensor:
		# query, key, value: (seq_len, bsz, d_model)
		seq_len, bsz, _ = query.size()
		k_seq_len = key.size(0)
		q = self.q_proj(query)
		k = self.k_proj(key)
		v = self.v_proj(value)
	
		# Reshape to (bsz, n_heads, seq_len, head_dim)
		q = q.view(-1, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
		k = k.view(-1, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
		v = v.view(-1, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

		if freqs_cis is not None:
			q = apply_rotary_emb(q, freqs_cis, start_pos, seq_len)
			k = apply_rotary_emb(k, freqs_cis, start_pos, k_seq_len)

		if kv_write_indices is not None:
			assert start_pos + k_seq_len <= self.max_seq_len, "Sequence length exceeds cache capacity"
			# k and v are of shape (bsz, n_heads, seq_len, head_dim).
			# Update along the seq_len dimension (dim=2).
			self.cache_k[:bsz].index_copy_(2, kv_write_indices, k)
			self.cache_v[:bsz].index_copy_(2, kv_write_indices, v)

			key = self.cache_k[:bsz, :, : start_pos + k_seq_len, :]
			value = self.cache_v[:bsz, :, : start_pos + k_seq_len, :]
		else:
			key = k
			value = v

		if attn_mask is not None:
			attn_mask = F._canonical_mask(
				mask=attn_mask,
				mask_name="attn_mask",
				other_type=None,
				other_name=None,
				target_type=query.dtype,
			).unsqueeze(1).unsqueeze(2)

		attn_output = F.scaled_dot_product_attention(
			q, key, value,
			attn_mask=attn_mask, dropout_p=self.p,
			scale=self.scaling, is_causal=is_causal
		)
		attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, bsz, self.d_model)

		output = self.out_proj(attn_output)
		return output
