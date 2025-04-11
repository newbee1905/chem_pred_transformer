import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from models.positional_encoding import precompute_freqs_cis
from models.transformer import EncoderLayer, DecoderLayer
from models.norm import RMSNorm

from utils import is_valid_smiles

class BART(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256,
		dropout: float = 0.1,
		norm_layer=RMSNorm,
		activation: str = "swiglu",
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

		self.freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=self.max_seq_len * 2, theta=10000.0)

		self.enc_layers = nn.ModuleList([
			EncoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_layer=norm_layer, activation=activation)
			for _ in range(n_layers)
		])

		self.dec_layers = nn.ModuleList([
			DecoderLayer(d_model, n_heads, d_ff, dropout, activation, norm_layer=norm_layer, activation=activation)
			for _ in range(n_layers)
		])

		self.token_fc = nn.Linear(d_model, vocab_size)

	def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
		# src: (batch, seq_len) -> (seq_len, batch)
		src = self.emb(src).transpose(0, 1)

		for layer in self.enc_layers:
			src = layer(src, src_mask, freqs_cis=self.freqs_cis)

		return src

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None
	) -> torch.Tensor:
		# tgt: (batch, seq_len) -> (seq_len, batch)
		tgt = self.emb(tgt).transpose(0, 1)

		for layer in self.dec_layers:
			tgt = layer(tgt, memory, tgt_mask, memory_mask, freqs_cis=self.freqs_cis)

		return tgt

	def generate(
		self, src_tokens: torch.Tensor, src_mask: torch.Tensor, sampler,
		max_length: int = 50, **sampler_kwargs
	) -> torch.Tensor:
		"""Generate full text using an external sampler."""
		memory = self.encode(src_tokens, src_mask)
		return sampler(self, memory, src_mask, max_length, **sampler_kwargs)

	def forward(
		self, src: torch.Tensor, tgt: torch.Tensor,
		src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
	) -> torch.Tensor:
		if self.freqs_cis.device != src.device:
			self.freqs_cis = self.freqs_cis.to(src.device)

		memory = self.encode(src_tokens, src_mask)
		decoder_output = self.decode(tgt_tokens, memory, tgt_mask, src_mask)
		logits = self.token_fc(decoder_output)

		return logits.transpose(0, 1)	# (batch, seq_len, vocab_size)

ATOM_LIST = ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

class BARTMT(BART):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256,
		dropout: float = 0.1,
		norm_layer=RMSNorm,
		activation: str = "swiglu",
	):
		super().__init__(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout, norm_layer, activation)

		self.weight_fc = nn.Sequential(
			nn.Linear(self.d_model, self.bart.config.d_model),
			nn.ReLU(),
			nn.Linear(self.d_model, invariant_dim)
		)

		self.atom_count_fc = nn.Sequential(
			nn.Linear(self.d_model, self.bart.config.d_model),
			nn.ReLU(),
			nn.Linear(self.bart.config.d_model, len(ATOM_LIST))
		)

	def forward(
		self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor,
		src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
		weight_target: torch.Tensor = None, atom_counts_target: torch.Tensor = None,
	) -> torch.Tensor:
		memory = self.encode(src_tokens, src_mask)

		pooled = memory.mean(dim=1)
		atom_counts_pred = self.atom_count_fc(pooled)
		weight_pred = self.weight_fc(pooled)

		if atom_counts_target is not None:
			atom_loss = F.mse_loss(atom_counts_pred, atom_counts_target)
			loss = loss + atom_loss

		if weight_target is not None:
			weight_loss = F.mse_loss(weight_pred, weight_target)
			loss = loss + weight_loss

		decoder_output = self.decode(tgt_tokens, memory, tgt_mask, src_mask)
		logits = self.token_fc(decoder_output)

		return logits.transpose(0, 1)	# (batch, seq_len, vocab_size)

# class BART(nn.Module):
# 	def __init__(
# 		self, vocab_size: int,
# 		d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
# 		d_ff: int = 2048, max_seq_len: int = 512,
# 		dropout: float = 0.1,
# 		norm_layer=nn.LayerNorm,
# 		activation: str = "swiglu",
# 	):
# 		super().__init__()
#
# 		self.vocab_size = vocab_size
# 		self.d_model = d_model
#
# 		self.emb = nn.Embedding(vocab_size, d_model)
#
# 		self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2)
#
# 		self.enc_layers = nn.ModuleList([
# 			EncoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer, activation=activation)
# 			for _ in range(n_layers)
# 		])
#
# 		self.dec_layers = nn.ModuleList([
# 			DecoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer, activation=activation)
# 			for _ in range(n_layers)
# 		])
#
# 		self.fc_out = nn.Linear(d_model, vocab_size)
# 		self.dropout = nn.Dropout(dropout)
#
# 	def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
# 		x = self.emb(src)
#
# 		for layer in self.enc_layers:
# 			x = layer(x, self.freqs_cis, src_mask)
#
# 		return x
#
# 	def decode(
# 		self, tgt: torch.Tensor, memory: torch.Tensor,
# 		tgt_mask: Optional[torch.Tensor] = None,
# 		memory_mask: Optional[torch.Tensor] = None,
# 	) -> torch.Tensor:
# 		x = self.emb(tgt)
#
# 		for layer in self.dec_layers:
# 			x, _ = layer(x, memory, self.freqs_cis, tgt_mask, memory_mask)
#
# 		return x
#
# 	def decode_incremental(
# 		self, tgt: torch.Tensor, memory: torch.Tensor,
# 		tgt_mask: Optional[torch.Tensor] = None,
# 		memory_mask: Optional[torch.Tensor] = None,
# 		caches: Optional[list] = None,
# 	) -> (torch.Tensor, list):
# 		x = self.emb(tgt)
#
# 		new_caches = []
# 		for i, layer in enumerate(self.dec_layers):
# 			layer_cache = caches[i] if caches is not None else None
# 			cur_pos = 0 if layer_cache is None or "cur_pos" not in layer_cache else layer_cache["cur_pos"]
# 			freqs_slice = self.freqs_cis[cur_pos : cur_pos + memory.shape[1]]
#
# 			x, layer_cache = layer(x, memory, freqs_slice, tgt_mask, memory_mask, cache=layer_cache)
# 			new_caches.append(layer_cache)
#
# 		return x, new_caches
#
# 	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
# 		_, seq_len = src.shape
#
# 		enc_out = self.encode(src, src_mask)
# 		dec_out = self.decode(tgt, enc_out, tgt_mask)
#
# 		out = self.fc_out(dec_out)
# 		return self.dropout(out)
#
# 	def generate_greedy(
# 		self,
# 		src: torch.Tensor,
# 		max_length: int = 256,
# 		bos_token_id: int = 0,
# 		eos_token_id = 1,
# 		src_mask: Optional[torch.Tensor] = None,
# 		top_k: int = 0,
# 	) -> torch.Tensor:
# 		device = src.device
# 		batch_size, src_seq_len = src.size()
# 		memory = self.encode(src, src_mask)
#
# 		caches = [{} for _ in self.dec_layers]
# 		generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
# 		done = torch.zeros(batch_size, dtype=torch.bool, device=device)
#
# 		for i in range(max_length):
# 			# new_token = generated[:, -1:].clone()
# 			# dec_out, caches = self.decode_incremental(new_token, memory, caches)
# 			# logits = self.fc_out(dec_out.squeeze(1))
# 			dec_out = self.decode(generated, memory)
# 			logits = self.fc_out(dec_out[:, -1, :])
#
# 			if top_k > 0:
# 				top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
#
# 				next_token_logits = torch.full_like(logits, float('-inf'))
# 				next_token_logits.scatter_(1, top_indices, top_values)
#
# 				probs = F.softmax(next_token_logits, dim=-1)
#
# 				next_tokens = torch.multinomial(probs, num_samples=1)
# 			else:
# 				next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
#
# 			next_tokens = next_tokens.masked_fill(done.unsqueeze(-1), eos_token_id)
# 			generated = torch.cat([generated, next_tokens], dim=1)
#
# 			done = done | (next_tokens.squeeze(-1) == eos_token_id)
#
# 			if done.all():
# 				break
#
# 		return generated
#
# 	def generate(
# 		self,
# 		src: torch.Tensor,
# 		max_length: int = 256,
# 		bos_token_id: int = 0,
# 		eos_token_id: int = 1,
# 		src_mask: Optional[torch.Tensor] = None,
# 		beam_width: int = 10,
# 		length_penalty: float = 0.7,
# 	) -> Tuple[torch.Tensor, torch.Tensor]:
# 		batch_size, src_seq_len = src.size()
# 		if batch_size != 1:
# 			raise NotImplementedError("Beam search is only implemented for batch size 1")
#
# 		device = src.device
#
# 		memory = self.encode(src, src_mask)
#
# 		beam_tokens = torch.full((1, max_length), eos_token_id, dtype=torch.long, device=device)
# 		beam_tokens[:, 0] = bos_token_id
# 		beam_scores = torch.zeros(1, device=device)
#
# 		finished = torch.zeros(1, dtype=torch.bool, device=device)
# 		current_length = 1
#
# 		vocab_size = self.fc_out.out_features 
#
# 		for t in range(1, max_length):
# 			num_beams = beam_tokens.size(0)
#
# 			current_seqs = beam_tokens[:, :current_length]
# 			memory_exp = memory.repeat(num_beams, 1, 1)
#
# 			dec_out = self.decode(current_seqs, memory_exp)
# 			logits = self.fc_out(dec_out[:, -1, :])
# 			log_probs = torch.log_softmax(logits, dim=-1) 
#
# 			# Set log-probs to -inf for all tokens except EOS.
# 			if finished.any():
# 				log_probs[finished] = -float("inf")
# 				log_probs[finished, eos_token_id] = 0.0
#
# 			total_scores = beam_scores.unsqueeze(1) + log_probs
# 			total_scores_flat = total_scores.view(-1)
#
# 			top_scores, top_indices = torch.topk(total_scores_flat, beam_width)
# 			new_beam_indices = top_indices // vocab_size
# 			new_token_indices = top_indices % vocab_size
#
# 			new_beam_tokens = beam_tokens[new_beam_indices].clone()
# 			new_beam_tokens[:, current_length] = new_token_indices
#
# 			new_finished = finished[new_beam_indices] | (new_token_indices == eos_token_id)
#
# 			# Apply length normalisation to beams that are finished
# 			for i in range(beam_width):
# 				if new_finished[i] and (current_length + 1 > src_seq_len):
# 					new_beam_scores_i = top_scores[i] / ((current_length + 1) ** length_penalty)
# 					top_scores[i] = new_beam_scores_i
#
# 			beam_tokens = new_beam_tokens
# 			beam_scores = top_scores
# 			finished = new_finished
# 			current_length += 1
#
# 			if finished.all():
# 				break
#
# 		return beam_tokens, beam_scores
#
# 	def sort_beam_candidates(self, ref_smi, gen_smi_candidates, scores, alpha=0.1, beta=10.0):
# 		candidates = []
# 		for i, smile in enumerate(gen_smi_candidates):
# 			length_penalty = max(0, abs(len(smile) - len(ref_smi)) - 2)
#
# 			valid_penalty = 0.0 if is_valid_smiles(smile) else 1.0
# 			# new_score = scores[i].item() - alpha * length_penalty - beta * valid_penalty
# 			new_score = scores[i].item() - alpha * length_penalty - beta * valid_penalty
#
# 			candidates.append((smile, new_score))
#
# 		candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
# 		return candidates_sorted
