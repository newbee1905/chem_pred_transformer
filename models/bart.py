import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from models.positional_encoding import precompute_freqs_cis
from models.norm import RMSNorm
from models.base import Base

from utils import is_valid_smiles

class BART(Base):
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
		super().__init__(
			vocab_size=vocab_size,
			d_model=d_model,
			n_heads=n_heads,
			n_layers=n_layers,
			d_ff=d_ff,
			max_seq_len=max_seq_len,
			max_batch_size=max_batch_size,
			dropout=dropout,
			use_layerscale=use_layerscale,
			norm_layer=norm_layer,
			activation=activation,
			aux_head=aux_head,
		)

		self.freqs_cis = precompute_freqs_cis(dim=self.head_dim, end=self.max_seq_len * 2, theta=theta)

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

	def forward(
		self, src: torch.Tensor, tgt: torch.Tensor,
		src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		if self.freqs_cis.device != src.device:
			self.freqs_cis = self.freqs_cis.to(src.device)

		return super().forward(src, tgt, src_mask, tgt_mask)
