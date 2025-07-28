import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Tuple, Optional

class FeedForward(nn.Module):
	"""Feedforward block with configurable activation.

	Supports:
	- 'swiglu': uses SiLU on the first half and multiplies with the second half.
	- 'geglu': uses GELU on the first half and multiplies with the second half.
	- 'gelu': standard feedforward with GELU.
	- 'silu': standard feedforward with SiLU.
	"""
	def __init__(
			self,
			d_model: int,
			d_ff: int,
			dropout: float = 0.1,
			activation: str = "SwiGLU",
		):
		super().__init__()

		self.activation = activation.lower()
		if self.activation not in ('swiglu', 'silu', 'geglu', 'gelu'):
			raise ValueError(f"Unknown activation type: {activation}")

		self.uses_gate = self.activation in ('swiglu', 'geglu')
		self.act_fn = F.silu if self.activation in ('swiglu', 'silu') else F.gelu

		# TODO: checking out parallel Linear from llama
		# fc_in can be column parallel
		# fc_out can be row parallel

		if self.activation in ('swiglu', 'geglu'):
			# default scaling down by 2/3 since normal 
			# d_ff is 4xd_model
			# Should be ~2.667 scalling now
			# based on Llama SwiGLU FeedForward
			# https://github.com/meta-llama/llama
			d_ff = int(2 * d_ff // 3)
			self.fc_in = nn.Linear(d_model, d_ff * 2)
		else:
			self.fc_in = nn.Linear(d_model, d_ff)

		self.fc_out = nn.Linear(d_ff, d_model) # can be row parallel

		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x_proj = self.fc_in(x)

		if self.activation in ('swiglu', 'geglu'):
			gate, x_proj = x_proj.chunk(2, dim=-1)
			x_proj = gate * self.act_fn(x_proj)
		else:
			x_proj = self.act_fn(x_proj)

		x = self.fc_out(self.dropout(x_proj))

		return x

class AttentionPooler(nn.Module):
	def __init__(self, d_model: int):
		super().__init__()
		self.attention_projector = nn.Linear(d_model, 1)

	def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		attention_scores = self.attention_projector(hidden_states)

		if mask is not None:
			mask_transposed = mask.transpose(0, 1).unsqueeze(-1)
			attention_scores = attention_scores.masked_fill(mask_transposed, float('-inf'))

		attention_weights = F.softmax(attention_scores, dim=0)

		pooled_output = (hidden_states * attention_weights).sum(dim=0)
		return pooled_output
