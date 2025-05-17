import math
import itertools
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from models.positional_encoding import SinusoidalPositionalEncoding
from models.base import Base

class Chemformer(Base):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256, max_batch_size: int = 256,
		dropout: float = 0.1,
		norm_layer=nn.LayerNorm,
		activation: str = "gelu",
		aux_head: bool = False,
	):
		super().__init__(
			vocab_size=vocab_size,
			d_model=d_model,
			n_heads=n_heads,
			n_layers=n_layers,
			d_ff=d_ff,
			max_seq_len=max_seq_len,
			max_batch_size=max_batch_size,
			dropout=dropout,
			use_layerscale=False,
			norm_layer=norm_layer,
			activation=activation,
			aux_head=aux_head,
		)

		self.pos_encoder = SinusoidalPositionalEncoding(d_model)

	def encode(
		self,
		src: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		# src: (batch, seq_len) -> (seq_len, batch)
		src = self.emb(src).transpose(0, 1)
		src = self.pos_encoder(src)

		for layer in self.enc_layers:
			src = layer(src, src_mask)

		return src

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
		kv_cache: bool = False,
		start_pos: int = 0,
	) -> torch.Tensor:
		# tgt_tokens: (batch, seq_len) -> (seq_len, batch)
		tgt = self.emb(tgt).transpose(0, 1)
		tgt = self.pos_encoder(tgt)

		for layer in self.dec_layers:
			tgt = layer(tgt, memory, tgt_mask, memory_mask, kv_cache=kv_cache, start_pos=start_pos)

		return tgt
