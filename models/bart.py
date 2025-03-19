import math
import itertools
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Optional

from models.utils import DyT, precompute_freqs_cis
from models.transformer import EncoderLayer, DecoderLayer

class BART(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12,
		n_enc_layers: int = 6, n_dec_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 1024,
		dropout: float = 0.1,
		norm_layer=nn.LayerNorm,
	):
		super().__init__()

		self.vocab_size = vocab_size
		self.d_model = d_model

		self.enc_emb = nn.Embedding(vocab_size, d_model)
		self.dec_emb = nn.Embedding(vocab_size, d_model)

		self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2)

		self.enc_layers = nn.ModuleList([
			EncoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer)
			for _ in range(n_enc_layers)
		])

		self.dec_layers = nn.ModuleList([
			DecoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer)
			for _ in range(n_dec_layers)
		])

		self.fc_out = nn.Linear(d_model, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.enc_emb(src)

		for layer in self.enc_layers:
			x = layer(x, self.freqs_cis, src_mask)

		return x

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None,
		memory_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		x = self.dec_emb(tgt)

		for layer in self.dec_layers:
			x = layer(x, memory, self.freqs_cis, tgt_mask, memory_mask)

		return x

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
		_, seq_len = src.shape

		enc_out = self.encode(src, src_mask)
		dec_out = self.decode(tgt, enc_out, tgt_mask)

		out = self.fc_out(dec_out)
		return self.dropout(out)

	def generate(
		self,
		src: torch.Tensor,
		max_length: int = 256,
		bos_token_id: int = 0,
		eos_token_id = 1,
		src_mask: Optional[torch.Tensor] = None,
		eos_boost: float = 5.0,
	) -> torch.Tensor:
		device = src.device
		batch_size, src_seq_len = src.size()
		memory = self.encode(src, src_mask)

		generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
		done = torch.zeros(batch_size, dtype=torch.bool, device=device)

		for i in range(max_length):
			dec_out = self.decode(generated, memory)
			logits = self.fc_out(dec_out[:, -1, :])

			if top_k > 0:
				top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)

				next_token_logits = torch.full_like(logits, float('-inf'))
				next_token_logits.scatter_(1, top_indices, top_values)

				probs = F.softmax(next_token_logits, dim=-1)

				next_tokens = torch.multinomial(probs, num_samples=1)
			else:
				next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

			next_tokens = next_tokens.masked_fill(done.unsqueeze(-1), eos_token_id)
			generated = torch.cat([generated, next_tokens], dim=1)

			done = done | (next_tokens.squeeze(-1) == eos_token_id)

			if done.all():
				break

		return generated
