import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import DyT, precompute_freqs_cis
from models.transformer import EncoderLayer, DecoderLayer

# Note: Based on https://arxiv.org/abs/2108.02446 and https://arxiv.org/abs/2004.04092
# https://github.com/seongminp/transformers-into-vaes
# https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/#pytorch-vae-implementation
class BARTVAE(nn.Module):
	def __init__(
		self, vocab_size: int,
		d_model: int = 768, n_heads: int = 12,
		n_enc_layers: int = 6, n_dec_layers: int = 6,
		d_ff: int = 3072, max_seq_len: int = 1024,
		dropout: float = 0.1,
		latent_dim: int = 128,
		norm_layer=nn.LayerNorm,
	):
		super().__init__()

		self.vocab_size = vocab_size
		self.d_model = d_model
		self.latent_dim = latent_dim

		self.enc_emb = nn.Embedding(vocab_size, d_model)
		self.dec_emb = nn.Embedding(vocab_size, d_model)

		self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2)

		self.enc_layers = nn.ModuleList([
			EncoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer)
			for _ in range(n_enc_layers)
		])

		self.dec_layers = nn.ModuleList([
			DecoderLayer(d_model, n_heads, d_ff, dropout, max_seq_len, norm_layer=norm_layer)
			for _ in range(n_dec_layers)
		])

		self.fc_out = nn.Linear(d_model, vocab_size)
		self.dropout = nn.Dropout(dropout)


		# Latent space transformations
		self.softplus = nn.Softplus()
		# self.fc_hidden_to_latten = nn.Linear(d_model * max_seq_len, latent_dim * 2)
		self.fc_hidden_to_latten = nn.Linear(d_model, latent_dim * 2)
		self.fc_latent_to_hidden = nn.Linear(latent_dim, d_model * max_seq_len)

	def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.enc_emb(src)

		for layer in self.enc_layers:
			x = layer(x, self.freqs_cis, src_mask)

		# TODO: may need to pool the output of encoder instead
		# x = x.view(x.size(0), -1)
		x = x.max(x.size(0), dim=1)
		x = self.fc_hidden_to_latten(x)

		mu, logvar = torch.chunk(x, 2, dim=-1)
		return mu, log_var

	def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = self.softplus(logvar) + eps
    scale_tril = torch.diag_embed(scale)

    dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    z = dist.rsample()

    return z, dist

	def decode(
		self, z: torch.Tensor, tgt: torch.Tensor,
		tgt_mask: Optional[torch.Tensor] = None,
		memory_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		memory = self.fc_latent_to_hidden(z)
		memory = memory.view(memory.size(0), -1, self.d_model)

		x = self.dec_emb(tgt)

		for layer in self.dec_layers:
			x = layer(x, memory, self.freqs_cis, tgt_mask, memory_mask)

		return x

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
		mu, log_var = self.encode(src, src_mask)
		z, dist = self.reparameterize(mu, log_var)
		dec_out = self.decode(tgt, z, tgt_mask)

		out = self.fc_out(dec_out)
		return self.dropout(out), z, dist
