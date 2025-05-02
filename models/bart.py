import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from models.positional_encoding import precompute_freqs_cis
from models.transformer import EncoderLayer, DecoderLayer
from models.norm import RMSNorm

from utils import is_valid_smiles

class BART(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12, n_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 256, max_batch_size: int = 256,
		dropout: float = 0.1,
		use_layerscale: bool = True,
		norm_layer=RMSNorm,
		activation: str = "swiglu",
		theta: float = 10000.0,
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

		self.freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=self.max_seq_len * 2, theta=theta)

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
			self.shared_proj = nn.Sequential(
				nn.Linear(d_model, d_ff),
				nn.SiLU(),
			)

			self.aux_heads = nn.ModuleDict({
				name: nn.Linear(d_ff, 1)
				for name in self.scalar_props
			})

	def encode(
		self,
		src: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		# src: (batch, seq_len) -> (seq_len, batch)
		src = self.emb(src).transpose(0, 1)

		for layer in self.enc_layers:
			src = layer(src, src_mask, freqs_cis=self.freqs_cis)

		return src

	def decode(
		self, tgt: torch.Tensor, memory: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
		kv_write_indices: Optional[torch.Tensor] = None,
		start_pos: int = 0,
	) -> torch.Tensor:
		# tgt: (batch, seq_len) -> (seq_len, batch)
		tgt = self.emb(tgt).transpose(0, 1)

		for layer in self.dec_layers:
			tgt = layer(
				tgt, memory, tgt_mask, memory_mask,
				freqs_cis=self.freqs_cis,
				kv_write_indices=kv_write_indices,
				start_pos=start_pos,
			)

		return tgt

	def generate(
		self, src: torch.Tensor, src_mask: torch.Tensor, sampler,
		max_length: int = 50, **sampler_kwargs
	) -> torch.Tensor:
		"""Generate full text using an external sampler."""
		if self.freqs_cis.device != src.device:
			self.freqs_cis = self.freqs_cis.to(src.device)

		memory = self.encode(src, src_mask)

		return sampler(self, memory, src_mask, max_length, **sampler_kwargs)

	def forward(
		self, src: torch.Tensor, tgt: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		if self.freqs_cis.device != src.device:
			self.freqs_cis = self.freqs_cis.to(src.device)

		memory = self.encode(src, src_mask)
		decoder_output = self.decode(tgt, memory, tgt_mask, src_mask)
		logits = self.token_fc(decoder_output)

		if not self.aux_head:
			return logits.transpose(0, 1)	# (batch, seq_len, vocab_size)

		tgt_memory = self.encode(tgt, tgt_mask)

		inp_pooled = memory.mean(dim=0)
		tgt_pooled = tgt_memory.mean(dim=0)

		inp_shared_proj = self.shared_proj(inp_pooled)
		tgt_shared_proj = self.shared_proj(tgt_pooled)

		aux_preds: Dict[str, torch.Tensor] = {}
		for name, head in self.aux_heads.items():
			aux_preds[f"react_{name}"] = head(inp_shared_proj).squeeze(-1)
			aux_preds[f"prod_{name}"]  = head(tgt_shared_proj).squeeze(-1)

		return logits.transpose(0, 1), aux_preds
