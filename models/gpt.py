import math
import itertools
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from typing import Optional

from models.utils import DyT, precompute_freqs_cis
from models.transformer import GPTDecoderLayer

class GPT(nn.Module):
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

		self.layers = nn.ModuleList([
			GPTDecoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer, activation=activation)
			for _ in range(n_layers)
		])

		self.fc_out = nn.Linear(d_model, vocab_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.emb(x)
		for layer in self.layers:
			x = layer(x, self.freqs_cis, src_mask)
		out = self.fc_out(x)
		return self.dropout(out)

	def generate(
		self,
		max_length: int = 256,
		bos_token_id: int = 0,
		eos_token_id: int = 1,
		top_k: int = 0,
		batch_size: int = 1,
	) -> torch.Tensor:
		device = next(self.parameters()).device
		generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
		done = torch.zeros(batch_size, dtype=torch.bool, device=device)

		for i in range(max_length):
			logits = self.forward(generated)[:, -1, :]

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

	def load_from_bart(self, bart_model: nn.Module) -> None:
		self.emb.weight.data.copy_(bart_model.emb.weight.data)
		
		for gpt_layer, bart_layer in zip(self.layers, bart_model.dec_layers):
			# Self attention
			gpt_layer.self_attn.q_proj.weight.data.copy_(bart_layer.self_attn.q_proj.weight.data)
			gpt_layer.self_attn.q_proj.bias.data.copy_(bart_layer.self_attn.q_proj.bias.data)
			
			gpt_layer.self_attn.k_proj.weight.data.copy_(bart_layer.self_attn.k_proj.weight.data)
			gpt_layer.self_attn.k_proj.bias.data.copy_(bart_layer.self_attn.k_proj.bias.data)
			
			gpt_layer.self_attn.v_proj.weight.data.copy_(bart_layer.self_attn.v_proj.weight.data)
			gpt_layer.self_attn.v_proj.bias.data.copy_(bart_layer.self_attn.v_proj.bias.data)
			

			# Self attention norm
			gpt_layer.self_attn_norm.weight.data.copy_(bart_layer.self_attn_norm.weight.data)
			# gpt_layer.self_attn_norm.bias.data.copy_(bart_layer.self_attn_norm.bias.data)

			gpt_layer.self_attn.out_proj.weight.data.copy_(bart_layer.self_attn.out_proj.weight.data)
			gpt_layer.self_attn.out_proj.bias.data.copy_(bart_layer.self_attn.out_proj.bias.data)

			if gpt_layer.self_attn_layer_scale is not None and bart_layer.self_attn_layer_scale is not None:
				gpt_layer.self_attn_layer_scale.data.copy_(bart_layer.self_attn_layer_scale.data)
			
			# Feedforward and Feedforward Norm
			gpt_layer.ff_norm.weight.data.copy_(bart_layer.ff_norm.weight.data)
			# gpt_layer.ff_norm.bias.data.copy_(bart_layer.ff_norm.bias.data)

			gpt_layer.ff.fc_in.weight.data.copy_(bart_layer.ff.fc_in.weight.data)
			gpt_layer.ff.fc_in.bias.data.copy_(bart_layer.ff.fc_in.bias.data)
			gpt_layer.ff.fc_out.weight.data.copy_(bart_layer.ff.fc_out.weight.data)

			if gpt_layer.ff_layer_scale is not None and bart_layer.ff_layer_scale is not None:
				gpt_layer.ff_layer_scale.data.copy_(bart_layer.ff_layer_scale.data)
			

		self.fc_out.weight.data.copy_(bart_model.fc_out.weight.data)
		self.fc_out.bias.data.copy_(bart_model.fc_out.bias.data)
