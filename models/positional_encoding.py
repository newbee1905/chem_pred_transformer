import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
	def __init__(self, d_model: int, max_len: int = 5000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(1)	# (max_len, 1, d_model)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (seq_len, batch, d_model)
		seq_len = x.size(0)
		return x + self.pe[:seq_len]

# RoPE code from gemma
# https://github.com/google/gemma_pytorch/blob/main/gemma/model.py
def precompute_freqs_cis(
	dim: int,
	end: int,
	theta: float = 10000.0,
	rope_scaling_factor: int = 1
) -> torch.Tensor:
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
	freqs = freqs/rope_scaling_factor

	t = torch.arange(end, device=freqs.device)

	freqs = torch.outer(t, freqs).float()
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

	return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
	x_complex = torch.view_as_complex(
		torch.stack(
			torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
			dim=-1,
		)
	)

	# freqs_cis = reshape_for_broadcast(freqs_cis, x_complex)
	freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension
	freqs_cis = freqs_cis.expand(x_complex.shape[0], x_complex.shape[1], -1, -1)  # Expand to matc

	x_out = torch.view_as_real(x_complex * freqs_cis).type_as(x)
	x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
	x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)

	return x_out
