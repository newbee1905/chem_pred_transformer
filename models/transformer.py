import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import xformers.ops as xops

from typing import Optional

from models.utils import DyT, FeedForward, apply_rotary_emb

# TODO: Implement MHA alternative with Convexifying attention and RWKV version to compare it with default one

class MultiHeadAttention(nn.Module):
	def __init__(self, d_model, n_heads, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
		
		self.d_model = d_model
		self.n_heads = n_heads
		self.d_k = d_model // n_heads
		
		# TODO: optimise with qkv_proj instead of 3 separate linear
		self.q_proj = nn.Linear(d_model, d_model)
		self.k_proj = nn.Linear(d_model, d_model)
		self.v_proj = nn.Linear(d_model, d_model)
		self.out_proj = nn.Linear(d_model, d_model)

		self.p = dropout
		# self.dropout = nn.Dropout(dropout)

		nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / (2 ** 0.5))
		nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / (2 ** 0.5))
		nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / (2 ** 0.5))
		nn.init.xavier_uniform_(self.out_proj.weight)
		nn.init.zeros_(self.out_proj.bias)
			
	def scaled_dot_product_attention(
		self, 
		Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
		mask: Optional[torch.Tensor] = None, is_causal: bool = False
	):
		# attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
		# if mask is not None:
		# 	attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

		# attn_probs = torch.softmax(attn_scores, dim=-1)
		# output = torch.matmul(attn_probs, V)
		# output = self.dropout(output)

		output = xops.memory_efficient_attention(
			Q, K, V, 
			p=self.p,
			attn_bias=None if not is_causal else xops.LowerTriangularMask(),
		)

		return output

	def split_heads(self, x):
		# return rearrange(x, 'b s (h d) -> b h s d', h=self.n_heads)
		return rearrange(x, 'b s (h d) -> b s h d', h=self.n_heads)
			
	def combine_heads(self, x):
		# return rearrange(x, 'b h s d -> b s (h d)', h=self.n_heads)
		return rearrange(x, 'b s h d -> b s (h d)', h=self.n_heads)
			
	def forward(
		self, 
  	Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, freqs_cis: torch.Tensor,
		mask: Optional[torch.Tensor] = None, is_causal: bool = False
	):
		Q = self.split_heads(self.q_proj(Q))
		K = self.split_heads(self.k_proj(K))
		V = self.split_heads(self.v_proj(V))

		Q, K = apply_rotary_emb(Q, K, freqs_cis=freqs_cis)
		
		attn_output = self.scaled_dot_product_attention(Q, K, V, mask, is_causal=is_causal)
		output = self.out_proj(self.combine_heads(attn_output))

		return output

class EncoderLayer(nn.Module):
	def __init__(
		self, d_model: int, n_heads: int, d_ff: int = 3072,
		dropout: float = 0.2, max_seq_len: int = 1024, use_layerscale: bool = True,
		norm_layer=nn.LayerNorm, activation="swiglu",
	):
		super().__init__()
		self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
		self.self_attn_norm = norm_layer(d_model)
		self.self_attn_dropout = nn.Dropout(dropout)
		self.attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.ff_norm = norm_layer(d_model)
		self.ff = FeedForward(d_model, d_ff, dropout, activation)
		self.ff_dropout = nn.Dropout(dropout)
		self.ff_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None
			
	def forward(self, src: torch.Tensor, freqs_cis: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
		norm_src = self.self_attn_norm(src)
		attn_out = self.self_attn(norm_src, norm_src, norm_src, freqs_cis, src_mask)
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
		dropout: float = 0.2, max_seq_len: int = 1024, use_layerscale: bool = True,
		norm_layer=nn.LayerNorm, activation="swiglu",
	):
		super().__init__()

		self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
		self.self_attn_norm = norm_layer(d_model)
		self.self_attn_dropout = nn.Dropout(dropout)
		self.self_attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
		self.cross_attn_norm = norm_layer(d_model)
		self.cross_attn_dropout = nn.Dropout(dropout)
		self.cross_attn_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.ff_norm = norm_layer(d_model)
		self.ff = FeedForward(d_model, d_ff, dropout, activation)
		self.ff_dropout = nn.Dropout(dropout)
		self.ff_layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4) if use_layerscale else None

		self.dropout = nn.Dropout(dropout)

	def forward(
		self, tgt: torch.Tensor, memory: torch.Tensor, freqs_cis: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
	):
		norm_tgt = self.self_attn_norm(tgt)
		self_attn_out = self.self_attn(norm_tgt, norm_tgt, norm_tgt, freqs_cis, tgt_mask, is_causal=True)
		self_attn_out = self.self_attn_dropout(self_attn_out)
		if self.self_attn_layer_scale is not None:
			tgt = tgt + self.self_attn_layer_scale * self_attn_out
		else:
			tgt = tgt + self_attn_out

		norm_tgt = self.cross_attn_norm(tgt)
		cross_attn_out = self.cross_attn(norm_tgt, memory, memory, freqs_cis, memory_mask)
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

class PreNormEncoderLayer(nn.TransformerEncoderLayer):
	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		att = self.norm1(src)
		att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
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
	):
		query = self.norm1(tgt)
		query = self.self_attn(
			query,
			query,
			query,
			attn_mask=tgt_mask,
			key_padding_mask=tgt_key_padding_mask,
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
		)[0]
		att = query + self.dropout2(att)

		out = self.norm3(att)
		out = self.linear2(self.dropout(self.activation(self.linear1(out))))
		out = att + self.dropout3(out)

		return out
