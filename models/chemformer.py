import math
import itertools
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from models.transformer import DecoderLayer, EncoderLayer
from models.positional_encoding import SinusoidalPositionalEncoding

class Chemformer(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256, max_batch_size: int = 256,
		dropout: float = 0.1,
		norm_layer=nn.LayerNorm,
		activation: str = "gelu",
	):
		super().__init__()

		self.vocab_size = vocab_size
		self.d_model = d_model
		self.max_seq_len = max_seq_len

		self.emb = nn.Embedding(vocab_size, d_model)
		self.dropout = nn.Dropout(dropout)

		self.n_heads = n_heads
		self.n_layers = n_layers
		self.d_ff = d_ff
		self.head_dim = d_model // n_heads

		self.pos_encoder = SinusoidalPositionalEncoding(d_model)

		self.enc_layers = nn.ModuleList([
			EncoderLayer(
				d_model,
				n_heads,
				d_ff,
				dropout, 
				use_layerscale=False, 
				norm_layer=norm_layer,
				activation=activation,
				max_seq_len=max_seq_len,
				max_batch_size=max_batch_size,
			)
			for _ in range(n_layers)
		])

		self.dec_layers = nn.ModuleList([
			DecoderLayer(
				d_model,
				n_heads,
				d_ff,
				dropout, 
				use_layerscale=False, 
				norm_layer=norm_layer,
				activation=activation,
				max_seq_len=max_seq_len,
				max_batch_size=max_batch_size,
			)
			for _ in range(n_layers)
		])

		self.token_fc = nn.Linear(d_model, vocab_size)

	def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
		# src: (batch, seq_len) -> (seq_len, batch)
		src = self.emb(src).transpose(0, 1)
		src = self.pos_encoder(src)

		for layer in self.enc_layers:
			src = layer(src, src_mask)

		return src

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None,
		kv_write_indices: Optional[torch.Tensor] = None,
		start_pos: int = 0,
	) -> torch.Tensor:
		# tgt_tokens: (batch, seq_len) -> (seq_len, batch)
		tgt = self.emb(tgt).transpose(0, 1)
		tgt = self.pos_encoder(tgt)

		for layer in self.dec_layers:
			tgt = layer(tgt, memory, tgt_mask, memory_mask)

		return tgt

	def generate(
		self, src: torch.Tensor, src_mask: torch.Tensor, sampler,
		max_length: int = 50, **sampler_kwargs
	) -> torch.Tensor:
		"""Generate full text using an external sampler."""
		memory = self.encode(src, src_mask)
		return sampler(self, memory, src_mask, max_length, **sampler_kwargs)

	def forward(
		self, src: torch.Tensor, tgt: torch.Tensor,
		src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
	) -> torch.Tensor:
		memory = self.encode(src, src_mask)
		decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)
		logits = self.token_fc(decoder_output)

		return logits.transpose(0, 1)	# (batch, seq_len, vocab_size)
