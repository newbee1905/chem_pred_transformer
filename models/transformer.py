import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import xformers.ops as xops

from typing import Optional

from models.positional_encoding import apply_rotary_emb
from models.mha import KVCacheMHA

class FeedForward(nn.Module):
	"""Feedforward block with configurable activation.

	Supports:
	- 'swiglu': uses SiLU on the first half and multiplies with the second half.
	- 'geglu': uses GELU on the first half and multiplies with the second half.
	- 'gelu': standard feedforward with GELU.
	- 'silu': standard feedforward with SiLU.
	"""
	def __init__(
		self,
		d_model: int,
		d_ff: int,
		dropout: float = 0.1,
		activation: str = "SwiGLU",
	):
		super().__init__()

		self.activation = activation.lower()
		if self.activation not in ('swiglu', 'silu', 'geglu', 'gelu'):
			raise ValueError(f"Unknown activation type: {activation}")

		self.uses_gate = self.activation in ('swiglu', 'geglu')
		self.act_fn = F.silu if self.activation in ('swiglu', 'silu') else F.gelu

		# TODO: checking out parallel Linear from llama
		# fc_in can be column parallel
		# fc_out can be row parallel

		if self.activation in ('swiglu', 'geglu'):
			# default scaling down by 2/3 since normal 
			# d_ff is 4xd_model
			# Should be ~2.667 scalling now
			# based on Llama SwiGLU FeedForward
			# https://github.com/meta-llama/llama
			d_ff = int(2 * d_ff // 3)
			self.fc_in = nn.Linear(d_model, d_ff * 2)
		else:
			self.fc_in = nn.Linear(d_model, d_ff)

		self.fc_out = nn.Linear(d_ff, d_model) # can be row parallel

		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_proj = self.fc_in(x)

		if self.activation in ('swiglu', 'geglu'):
			gate, x_proj = x_proj.chunk(2, dim=-1)
			x_proj = gate * self.act_fn(x_proj)
		else:
			x_proj = self.act_fn(x_proj)

		x = self.fc_out(self.dropout(x_proj))

		return x


class EncoderLayer(nn.Module):
	def __init__(
		self, d_model: int, n_heads: int, d_ff: int = 3072,
		dropout: float = 0.2, use_layerscale: bool = True,
		norm_layer=nn.LayerNorm, activation="swiglu",
		max_seq_len: int = 256, max_batch_size: int = 256,
	):
		super().__init__()
		self.self_attn = KVCacheMHA(d_model, n_heads, dropout, max_seq_len, max_batch_size)
		self.self_attn_norm = norm_layer(d_model)
		self.self_attn_dropout = nn.Dropout(dropout)
		self.attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.ff_norm = norm_layer(d_model)
		self.ff = FeedForward(d_model, d_ff, dropout, activation)
		self.ff_dropout = nn.Dropout(dropout)
		self.ff_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None
			
	def forward(
		self, src: torch.Tensor, src_mask: torch.Tensor = None,
		freqs_cis: Optional[torch.Tensor] = None, start_pos: int = 0,
		**kwargs,
	):
		norm_src = self.self_attn_norm(src)
		attn_out = self.self_attn(
			norm_src, norm_src, norm_src, src_mask,
			start_pos=start_pos, freqs_cis=freqs_cis,
			**kwargs,
		)
		attn_out = self.self_attn_dropout(attn_out)

		if self.attn_layer_scale is not None:
			src = src + self.attn_layer_scale * attn_out
		else:
			src = src + attn_out

		norm_src = self.ff_norm(src)
		ff_out = self.ff(norm_src)
		ff_out = self.ff_dropout(ff_out)

		if self.ff_layer_scale is not None:
			src = src + self.ff_layer_scale * ff_out
		else:
			src = src + ff_out

		return src

class DecoderLayer(nn.Module):
	def __init__(
		self, d_model: int, n_heads: int, d_ff: int = 3072,
		dropout: float = 0.2, use_layerscale: bool = True,
		norm_layer=nn.LayerNorm, activation="swiglu",
		max_seq_len: int = 256, max_batch_size: int = 256,
	):
		super().__init__()

		self.self_attn = KVCacheMHA(d_model, n_heads, dropout, max_seq_len, max_batch_size)
		self.self_attn_norm = norm_layer(d_model)
		self.self_attn_dropout = nn.Dropout(dropout)
		self.self_attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.cross_attn = KVCacheMHA(d_model, n_heads, dropout, max_seq_len, max_batch_size)
		self.cross_attn_norm = norm_layer(d_model)
		self.cross_attn_dropout = nn.Dropout(dropout)
		self.cross_attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.ff_norm = norm_layer(d_model)
		self.ff = FeedForward(d_model, d_ff, dropout, activation)
		self.ff_dropout = nn.Dropout(dropout)
		self.ff_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.dropout = nn.Dropout(dropout)

	def forward(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
		freqs_cis: Optional[torch.Tensor] = None,
		start_pos: int = 0, **kwargs,
	):
		norm_tgt = self.self_attn_norm(tgt)
		self_attn_out = self.self_attn(
			norm_tgt, norm_tgt, norm_tgt,
			tgt_mask, is_causal=True, 
			start_pos=start_pos, freqs_cis=freqs_cis,
			**kwargs,
		)
		self_attn_out = self.self_attn_dropout(self_attn_out)
		if self.self_attn_layer_scale is not None:
			tgt = tgt + self.self_attn_layer_scale * self_attn_out
		else:
			tgt = tgt + self_attn_out

		norm_tgt = self.cross_attn_norm(tgt)
		cross_attn_out = self.cross_attn(
			norm_tgt, memory, memory, memory_mask,
			start_pos=start_pos, freqs_cis=freqs_cis,
		)
		cross_attn_out = self.cross_attn_dropout(cross_attn_out)
		if self.cross_attn_layer_scale is not None:
			tgt = tgt + self.cross_attn_layer_scale * cross_attn_out
		else:
			tgt = tgt + cross_attn_out

		norm_tgt = self.ff_norm(tgt)
		ff_out = self.ff(norm_tgt)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			tgt = tgt + self.ff_layer_scale * ff_out
		else:
			tgt = tgt + ff_out

		return tgt

class GPTDecoderLayer(nn.Module):
	def __init__(
		self, d_model: int, n_heads: int, d_ff: int = 3072,
		dropout: float = 0.2, max_seq_len: int = 1024, use_layerscale: bool = True,
		norm_layer=nn.LayerNorm, activation="swiglu",
	):
		super().__init__()
		self.self_attn = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
		self.self_attn_norm = norm_layer(d_model)
		self.self_attn_dropout = nn.Dropout(dropout)
		self.self_attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.ff_norm = norm_layer(d_model)
		self.ff = FeedForward(d_model, d_ff, dropout, activation)
		self.ff_dropout = nn.Dropout(dropout)
		self.ff_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None
			
	def forward(self, src: torch.Tensor, freqs_cis: torch.Tensor, src_mask: Optional[torch.Tensor] = None, cache: Optional[dict] = None):
		norm_src = self.self_attn_norm(src)
		attn_out, cache = self.self_attn(norm_src, norm_src, norm_src, freqs_cis, src_mask, is_causal=True, cache=cache)
		attn_out = self.self_attn_dropout(attn_out)
		if self.self_attn_layer_scale is not None:
			src = src + self.self_attn_layer_scale * attn_out
		else:
			src = src + attn_out

		norm_src = self.ff_norm(src)
		ff_out = self.ff(norm_src)
		ff_out = self.ff_dropout(ff_out)
		if self.ff_layer_scale is not None:
			src = src + self.ff_layer_scale * ff_out
		else:
			src = src + ff_out

		return src, cache

class PreNormEncoderLayer(nn.TransformerEncoderLayer):
	def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
		att = self.norm1(src)
		att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal)[0]
		att = src + self.dropout1(att)

		out = self.norm2(att)
		out = self.linear2(self.dropout(self.activation(self.linear1(out))))
		out = att + self.dropout2(out)
		return out


class PreNormDecoderLayer(nn.TransformerDecoderLayer):
	def forward(
		self,
		tgt,
		memory,
		tgt_mask=None,
		memory_mask=None,
		tgt_key_padding_mask=None,
		memory_key_padding_mask=None,
		tgt_is_causal=False,
		memory_is_causal=False,
	):
		query = self.norm1(tgt)
		query = self.self_attn(
			query,
			query,
			query,
			attn_mask=tgt_mask,
			key_padding_mask=tgt_key_padding_mask,
			is_causal=tgt_is_causal,
		)[0]
		query = tgt + self.dropout1(query)

		# Context attention block
		att = self.norm2(query)
		att = self.multihead_attn(
			att,
			memory,
			memory,
			attn_mask=memory_mask,
			key_padding_mask=memory_key_padding_mask,
			is_causal=memory_is_causal,
		)[0]
		att = query + self.dropout2(att)

		out = self.norm3(att)
		out = self.linear2(self.dropout(self.activation(self.linear1(out))))
		out = att + self.dropout3(out)

		return out
