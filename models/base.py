import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from models.transformer import EncoderLayer, DecoderLayer

from utils import is_valid_smiles

class Base(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256, max_batch_size: int = 256,
		dropout: float = 0.1,
		use_layerscale: bool = True,
		norm_layer=nn.LayerNorm,
		activation: str = "gelu",
		aux_head: bool = False,
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

		self.enc_layers = nn.ModuleList([
			EncoderLayer(
				d_model,
				n_heads,
				d_ff,
				dropout,
				use_layerscale=use_layerscale,
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
				use_layerscale=use_layerscale,
				norm_layer=norm_layer,
				activation=activation,
				max_seq_len=max_seq_len,
				max_batch_size=max_batch_size,
			)
			for _ in range(n_layers)
		])

		self.token_fc = nn.Linear(d_model, vocab_size)

		self.aux_head = aux_head

		self.scalar_props = [
			'MolWt', 'LogP', 'TPSA',
			'NumHDonors', 'NumHAcceptors',
			'NumRotatableBonds', 'RingCount'
		]

		if self.aux_head:
			bottleneck = d_ff // 2

			self.pool_proj = nn.Linear(d_model, 1)

			self.shared_proj = nn.Sequential(
				nn.Linear(d_model, bottleneck),
				norm_layer(bottleneck),
				nn.SiLU(),
				nn.Dropout(dropout),
			)

			self.aux_out = nn.Sequential(
				nn.Linear(bottleneck, bottleneck),
				nn.SiLU(),
				nn.Dropout(dropout),
				nn.Linear(bottleneck, len(self.scalar_props)),
			)
			self.aux_skip = nn.Linear(bottleneck, len(self.scalar_props), bias=False)
			self.aux_gate = nn.Parameter(torch.zeros(len(self.scalar_props)))

			self.aux_logvars = nn.Parameter(torch.zeros(len(self.scalar_props)))

	def encode(
		self,
		src: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		raise NotImplementedError

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
		# kv_write_indices: Optional[torch.Tensor] = None,
		kv_cache: bool = False,
		start_pos: int = 0,
	) -> torch.Tensor:
		raise NotImplementedError


	def clear_cache(self):
		for layer in self.enc_layers:
			layer.self_attn.clear_cache()

		for layer in self.dec_layers:
			layer.self_attn.clear_cache()
			layer.cross_attn.clear_cache()

	def generate(
		self, src: torch.Tensor, src_mask: torch.Tensor, sampler,
		max_length: int = 50, 
		memory_override: torch.Tensor | None = None,
		**sampler_kwargs
	) -> torch.Tensor:
		"""Generate full text using an external sampler."""
		self.clear_cache()
		if memory_override is None:
			assert src is not None, "src required when memory not supplied"
			memory = self.encode(src, src_mask)
		else:
			memory = memory_override

		return sampler(self, memory, src_mask, max_length, **sampler_kwargs)

	def attn_pool(
		self,
		hidden: torch.Tensor,
		mask: Optional[torch.Tensor],
	) -> torch.Tensor:
		scores = self.pool_proj(hidden).squeeze(-1)
		if mask is not None:
			scores = scores.masked_fill(mask.transpose(0,1), float('-inf'))

		weights = F.softmax(scores, dim=0)

		return (hidden * weights.unsqueeze(-1)).sum(dim=0)


	def forward(
		self, src: torch.Tensor, tgt: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		memory = self.encode(src, src_mask)
		decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)
		logits = self.token_fc(decoder_output)

		if not self.aux_head:
			return logits.transpose(0, 1)	# (batch, seq_len, vocab_size)

		# tgt_memory = self.encode(tgt, tgt_mask)

		inp_pooled = self.attn_pool(memory, src_mask)
		# tgt_pooled = self.attn_pool(tgt_memory, tgt_mask)

		inp_shared_proj = self.shared_proj(inp_pooled)
		# tgt_shared_proj = self.shared_proj(tgt_pooled)

		inp_aux_out = self.aux_out(inp_shared_proj)
		inp_aux_skip = self.aux_skip(inp_shared_proj)

		# tgt_aux_out = self.aux_out(tgt_shared_proj)
		# tgt_aux_skip = self.aux_skip(tgt_shared_proj)

		gate = torch.sigmoid(self.aux_gate)
		aux_react = gate * inp_aux_out + (1 - gate) * inp_aux_skip
		# aux_prod = gate * tgt_aux_out + (1 - gate) * tgt_aux_skip

		aux_preds: Dict[str, torch.Tensor] = {}
		for i, name in enumerate(self.scalar_props):
			aux_preds[f"react_{name}"] = aux_react[:, i]
			# aux_preds[f"prod_{name}"] = aux_prod[:, i]

		return logits.transpose(0, 1), aux_preds

	def evaluate_actions(
		self, memory: torch.Tensor, src_mask: torch.Tensor, tgt_tokens: torch.Tensor, pad_token_id: int
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Evaluates the log-probabilities and entropy of a given sequence of actions.
		"""
		decoder_input = tgt_tokens[:, :-1]
		labels = tgt_tokens[:, 1:]

		decoder_output = self.decode(tgt=decoder_input, memory=memory, memory_mask=src_mask)
		logits = self.token_fc(decoder_output).transpose(0, 1) # Shape: [Batch, SeqLen-1, VocabSize]

		log_probs_dist = F.log_softmax(logits, dim=-1)
		gathered_log_probs = torch.gather(log_probs_dist, 2, labels.unsqueeze(-1)).squeeze(-1)
		
		probs_dist = torch.exp(log_probs_dist)
		entropy_per_token = -(probs_dist * log_probs_dist).sum(dim=-1)

		non_pad_mask = (labels != pad_token_id)
		gathered_log_probs = gathered_log_probs * non_pad_mask
		entropy_per_token = entropy_per_token * non_pad_mask

		sequence_log_probs = gathered_log_probs.sum(dim=1)
		sequence_entropy = entropy_per_token.sum(dim=1) / (non_pad_mask.sum(dim=1).float() + 1e-9)

		return sequence_log_probs, sequence_entropy, decoder_output
