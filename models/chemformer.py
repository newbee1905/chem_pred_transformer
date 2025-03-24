import math
import itertools
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Optional

from models.utils import DyT, precompute_freqs_cis
from models.transformer import PreNormEncoderLayer, PreNormDecoderLayer

class Chemformer(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
		d_ff: int = 2048, max_seq_len: int = 512,
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
		self.register_buffer("pos_emb", self._positional_embs())

		self.encoder = nn.TransformerEncoder(
			PreNormEncoderLayer(d_model, n_heads, d_ff, dropout, activation),
			n_layers,
			norm=nn.LayerNorm(d_model),
		)

		self.decoder = nn.TransformerDecoder(
			PreNormDecoderLayer(d_model, n_heads, d_ff, dropout, activation),
			n_layers,
			norm=nn.LayerNorm(d_model),
		)

		self.token_fc = nn.Linear(d_model, vocab_size)

	def _construct_input(self, token_ids, sentence_masks=None):
		token_embs = self.emb(token_ids)  # (batch, seq_len, d_model)
		token_embs = token_embs * math.sqrt(self.d_model)

		seq_len = token_ids.size(1)

		positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)

		embs = token_embs + positional_embs
		return self.dropout(embs).transpose(0, 1)

	def _positional_embs(self):
		encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
		encs = 10000**encs
		encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]

		encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
		encs = torch.stack(encs)

		return encs

	def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self._construct_input(src)
		x = self.encoder(x)

		return x

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None,
		memory_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		x = self._construct_input(tgt)
		x = self.decoder(x, memory)

		return x.transpose(0, 1)

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
		_, seq_len = src.shape

		enc_out = self.encode(src, src_mask)
		dec_out = self.decode(tgt, enc_out, tgt_mask)

		out = self.token_fc(dec_out)
		return out

	def generate(
		self,
		src: torch.Tensor,
		max_length: int = 256,
		bos_token_id: int = 0,
		eos_token_id = 1,
		src_mask: Optional[torch.Tensor] = None,
		top_k: int = 0,
	) -> torch.Tensor:
		device = src.device
		batch_size, src_seq_len = src.size()
		memory = self.encode(src, src_mask)

		generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
		done = torch.zeros(batch_size, dtype=torch.bool, device=device)

		for i in range(max_length):
			dec_out = self.decode(generated, memory)
			logits = self.token_fc(dec_out[:, -1, :])

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
