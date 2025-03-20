import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from models.vae import BARTVAE
from tokenizer import SMILESTokenizer

class LambdaScheduler:
	def __init__(self, start_lambda=0.0, max_lambda=1.0, warmup_epochs=10):
		self.start_lambda = start_lambda
		self.max_lambda = max_lambda
		self.warmup_epochs = warmup_epochs

	def get_lambda(self, epoch):
		return min(self.max_lambda, self.start_lambda + (self.max_lambda - self.start_lambda) * (epoch / self.warmup_epochs))


class BARTVAEModel(pl.LightningModule):
	def __init__(self, model: BARTVAE, tokenizer: SMILESTokenizer, lambda_warmup=10):
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
		self.kl_scheduler = LambdaScheduler(warmup_epochs=lambda_warmup)

		self.val_elbo = []
		self.val_kl = []
		self.val_recon_loss = []
		self.test_elbo = []
		self.test_kl = []
		self.test_recon_loss = []

	def forward(self, src, tgt, src_mask, tgt_mask):
		return self.model(src, tgt, src_mask, tgt_mask)

	def compute_loss(self, logits, target, z, dist, epoch):
		recon_loss = self.loss_fn(
			logits[:, 1:, :].contiguous().view(-1, logits.size(-1)),
			target.contiguous().view(-1)
		)
		# Caclulating KL Loss based on this
		# https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch
		std_normal = torch.distributions.MultivariateNormal(
			torch.zeros_like(z, device=z.device),
			scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
		)
		kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

		# Annealed λ
		kl_weight = self.kl_scheduler.get_lambda(epoch)
		elbo = recon_loss + kl_weight * kl_loss

		return elbo, recon_loss, kl_loss

	def training_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits, z, dist = self(src, decoder_input, mask, mask)
		elbo, recon_loss, kl_loss = self.compute_loss(logits, target, z, dist, self.current_epoch)

		self.log("train_elbo", elbo, prog_bar=True)
		self.log("train_kl", kl_loss, prog_bar=True)
		self.log("train_recon_loss", recon_loss, prog_bar=True)

		return elbo

	def validation_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits, z, dist = self(src, decoder_input, mask, mask)
		elbo, recon_loss, kl_loss = self.compute_loss(logits, target, z, dist, self.current_epoch)

		self.val_elbo.append(elbo)
		self.val_kl.append(kl_loss)
		self.val_recon_loss.append(recon_loss)

		self.log("val_elbo", elbo, prog_bar=True)
		self.log("val_kl", kl_loss, prog_bar=True)
		self.log("val_recon_loss", recon_loss, prog_bar=True)

	def on_validation_epoch_end(self):
		avg_elbo = torch.stack(self.val_elbo).mean()
		avg_kl = torch.stack(self.val_kl).mean()
		avg_recon_loss = torch.stack(self.val_recon_loss).mean()

		self.log_dict({
			"val_elbo": avg_elbo,
			"val_kl": avg_kl,
			"val_recon_loss": avg_recon_loss,
		}, prog_bar=True)

		self.val_elbo.clear()
		self.val_kl.clear()
		self.val_recon_loss.clear()

	def test_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits, z, dist = self(src, decoder_input, mask, mask)
		elbo, recon_loss, kl_loss = self.compute_loss(logits, target, z, dist, self.current_epoch)

		self.test_elbo.append(elbo)
		self.test_kl.append(kl_loss)
		self.test_recon_loss.append(recon_loss)

		self.log("test_elbo", elbo, prog_bar=True)
		self.log("test_kl", kl_loss, prog_bar=True)
		self.log("test_recon_loss", recon_loss, prog_bar=True)

	def on_test_epoch_end(self):
		avg_elbo = torch.stack(self.test_elbo).mean()
		avg_kl = torch.stack(self.test_kl).mean()
		avg_recon_loss = torch.stack(self.test_recon_loss).mean()

		self.log_dict({
			"test_elbo": avg_elbo,
			"test_kl": avg_kl,
			"test_recon_loss": avg_recon_loss,
		}, prog_bar=True)

		self.test_elbo.clear()
		self.test_kl.clear()
		self.test_recon_loss.clear()

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=5e-5)
