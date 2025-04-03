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

from utils import is_valid_smiles

class BART(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
		d_ff: int = 2048, max_seq_len: int = 512,
		dropout: float = 0.1,
		norm_layer=nn.LayerNorm,
		activation: str = "swiglu",
	):
		super().__init__()

		self.vocab_size = vocab_size
		self.d_model = d_model

		self.emb = nn.Embedding(vocab_size, d_model)

		self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2)

		self.enc_layers = nn.ModuleList([
			EncoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer, activation=activation)
			for _ in range(n_layers)
		])

		self.dec_layers = nn.ModuleList([
			DecoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer, activation=activation)
			for _ in range(n_layers)
		])

		self.fc_out = nn.Linear(d_model, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.emb(src)

		for layer in self.enc_layers:
			x = layer(x, self.freqs_cis, src_mask)

		return x

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None,
		memory_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		x = self.emb(tgt)

		for layer in self.dec_layers:
			x, _ = layer(x, memory, self.freqs_cis, tgt_mask, memory_mask)

		return x

	def decode_incremental(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None,
		memory_mask: Optional[torch.Tensor] = None,
		caches: Optional[list] = None,
	) -> (torch.Tensor, list):
		x = self.emb(tgt)

		new_caches = []
		for i, layer in enumerate(self.dec_layers):
			layer_cache = caches[i] if caches is not None else None
			cur_pos = 0 if layer_cache is None or "cur_pos" not in layer_cache else layer_cache["cur_pos"]
			freqs_slice = self.freqs_cis[cur_pos : cur_pos + memory.shape[1]]

			x, layer_cache = layer(x, memory, freqs_slice, tgt_mask, memory_mask, cache=layer_cache)
			new_caches.append(layer_cache)

		return x, new_caches

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
		_, seq_len = src.shape

		enc_out = self.encode(src, src_mask)
		dec_out = self.decode(tgt, enc_out, tgt_mask)

		out = self.fc_out(dec_out)
		return self.dropout(out)

	def generate_greedy(
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

		caches = [{} for _ in self.dec_layers]
		generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
		done = torch.zeros(batch_size, dtype=torch.bool, device=device)

		for i in range(max_length):
			# new_token = generated[:, -1:].clone()
			# dec_out, caches = self.decode_incremental(new_token, memory, caches)
			# logits = self.fc_out(dec_out.squeeze(1))
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

	def generate(
		self,
		src: torch.Tensor,
		max_length: int = 256,
		bos_token_id: int = 0,
		eos_token_id = 1,
		src_mask: Optional[torch.Tensor] = None,
		beam_width: int = 15,
		length_penalty: float = 0.7,
	) -> torch.Tensor:
		device = src.device
		batch_size, src_seq_len = src.size()
		if batch_size != 1:
			raise NotImplementedError("Beam search is only implemented for batch size 1")

		memory = self.encode(src, src_mask)
		beams = [([bos_token_id], 0.0)]

		for _ in range(max_length):
			new_beams = []

			for seq, score in beams:
				if seq[-1] == eos_token_id:
					penalised_score = score / (len(seq) ** length_penalty) if len(seq) > src_seq_len else score
					new_beams.append((seq, score))
					continue

				input_seq = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
				dec_out = self.decode(input_seq, memory)
				logits = self.fc_out(dec_out[:, -1, :])

				log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
				top_log_probs, top_indices = torch.topk(log_probs, beam_width)

				for i in range(beam_width):
					new_seq = seq + [top_indices[i].item()]
					new_score = score + top_log_probs[i].item()
					new_beams.append((new_seq, new_score))

			beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
			if all(seq[-1] == eos_token_id for seq, _ in beams):
				break

		return [b for b, _ in beams], [s for _, s in beams]

	def sort_beam_candidates(self, ref_smi, gen_smi_candidates, scores, alpha=0.1, beta=10.0):
		candidates = []
		for i, smile in enumerate(gen_smi_candidates):
			length_penalty = max(0, abs(len(smile) - len(ref_smi)) - 2)

			valid_penalty = 0.0 if is_valid_smiles(smile) else 1.0
			# new_score = scores[i].item() - alpha * length_penalty - beta * valid_penalty
			new_score = scores[i] - alpha * length_penalty - beta * valid_penalty

			candidates.append((smile, new_score))

		candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
		return candidates_sorted
