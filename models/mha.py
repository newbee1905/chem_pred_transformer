import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import xformers.ops as xops

import models.positional_encoding from apply_rotary_emb

class KVCacheMHA(nn.Module):
	def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
		super().__init__()
		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim must be divisible by n_heads")

		self.embed_dim = embed_dim
		self.n_heads = n_heads
		self.head_dim = embed_dim // n_heads
		self.scaling = self.head_dim ** -0.5

		self.q_proj = nn.Linear(embed_dim, embed_dim)
		self.k_proj = nn.Linear(embed_dim, embed_dim)
		self.v_proj = nn.Linear(embed_dim, embed_dim)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		self.p = dropout

	def forward(
		self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
		attn_mask: torch.Tensor = None, is_causal: bool = False,
		kv_cache: tuple = None,	kv_write_indices: torch.Tensor = None,
		freqs_cis: torch.Tensor = None,
		**kwargs,
	) -> torch.Tensor:
		# query, key, value: (seq_len, bsz, embed_dim)
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
				q = apply_rotary_emb(q, freqs_cis=freqs_cis)
				k = apply_rotary_emb(k, freqs_cis=freqs_cis)

		if kv_cache is not None and kv_write_indices is not None:
			k_cache, v_cache = kv_cache
			# k and v are of shape (bsz, n_heads, seq_len, head_dim).
			# Update along the seq_len dimension (dim=2).
			k_cache.index_copy_(2, kv_write_indices, k)
			v_cache.index_copy_(2, kv_write_indices, v)
			key = k_cache
			value = v_cache

		if attn_mask is not None:
			attn_mask = F._canonical_mask(
				mask=attn_mask,
				mask_name="attn_mask",
				other_type=None,
				other_name=None,
				target_type=query.dtype,
			).unsqueeze(1).unsqueeze(2)
			attn_mask = attn_mask.expand(bsz, self.n_heads, seq_len, k_seq_len)


		if is_causal:
			if attn_mask is None:
				attn_mask = xops.LowerTriangularMask()
				attn_mask = attn_mask.materialize(shape=(bsz, self.n_heads, seq_len, k_seq_len), device=q.device)
			else:
				attn_mask = xops.fmha.attn_bias.LowerTriangularMaskWithTensorBias(attn_mask)
				attn_mask = attn_mask.materialize(shape=(bsz, self.n_heads, seq_len, k_seq_len), device=q.device)

		attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p, scale=self.scaling)
		attn_output = attn_output.permute(2, 0, 1, 3).reshape(seq_len, bsz, self.embed_dim)

		output = self.out_proj(attn_output)
		return output
