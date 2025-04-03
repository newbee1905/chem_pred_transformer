import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Tuple

# TODO: recheck the paper again cause currently
# it is slower in training compare to RMSNorm
class DyT(nn.Module):
	def __init__(self, C, init_alpha=0.5):
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
		self.gamma = nn.Parameter(torch.ones(C))
		self.beta = nn.Parameter(torch.zeros(C))

	def forward(self, x):
		x = torch.tanh(self.alpha * x)

		return self.gamma * x + self.beta

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device='cpu').float() / dim))
	t = torch.arange(end, device=inv_freq.device).float()
	freqs = torch.outer(t, inv_freq)
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

	return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
		f"freqs_cis shape {freqs_cis.shape} needs to be {(x.shape[1], x.shape[-1])}"
	)

	ndim = x.ndim
	shape = [1] * x.ndim
	shape[1] = x.shape[1]
	shape[-1] = x.shape[-1]

	return freqs_cis.view(*shape)

def apply_rotary_emb(
	xq: torch.Tensor,
	xk: torch.Tensor,
	freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	xq_ = rearrange(xq.float(), "... (n two) -> ... n two", two=2)
	xk_ = rearrange(xk.float(), "... (n two) -> ... n two", two=2)
	xq_complex = torch.view_as_complex(xq_)
	xk_complex = torch.view_as_complex(xk_)

	freqs_cis_q = reshape_for_broadcast(freqs_cis[: xq.shape[1]], xq_complex).to(xq.device)
	freqs_cis_k = reshape_for_broadcast(freqs_cis[: xk.shape[1]], xk_complex).to(xk.device)

	xq_rot = xq_complex * freqs_cis_q
	xk_rot = xk_complex * freqs_cis_k

	xq_out = torch.view_as_real(xq_rot).flatten(3)
	xk_out = torch.view_as_real(xk_rot).flatten(3)

	return xq_out.type_as(xq), xk_out.type_as(xk)
