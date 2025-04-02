import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from transformers import PreTrainedTokenizerFast
from rdkit import Chem

import torchmetrics
from torchmetrics.text.bleu import BLEUScore

from einops import rearrange, repeat

from typing import Optional, Tuple

from models.bart import BART
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from metrics import SMILESEvaluationMetric

from transformers import get_cosine_schedule_with_warmup

class BARTModel(pl.LightningModule): 
	def __init__(self, model: BART, tokenizer: SMILESTokenizer | ChemformerTokenizer, max_length: int = 256, mode: str = "pretrain"):
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

		# Track accuracy metrics
		self.val_top1_acc = []
		self.val_top5_acc = []
		self.test_top1_acc = []
		self.test_top5_acc = []

		self.smiles_metric = SMILESEvaluationMetric()
		self.max_length = max_length

		self.mode = "pretrain"

	def forward(self, src, tgt, src_mask = None, tgt_mask = None):
		return self.model(src, tgt, src_mask, tgt_mask)

	def _calc_loss(self, tgt: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
		# Remove the first token (BOS) from both target and logits.
		token_output = rearrange(logits[:, 1:, :], 'b s v -> s b v')
		target = rearrange(tgt[:, 1:], 'b s -> s b')

		token_output = token_output.reshape(-1, token_output.size(-1)).float()
		target = target.reshape(-1)
		loss = self.loss_fn(token_output, target)

		return loss

	def _calc_token_acc(self, tgt: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute token-level top-1 and top-5 accuracy over the entire sequence,
		ignoring pad tokens.
		"""
		pad_id = self.tokenizer.pad_token_id

		_, pred_ids = torch.max(logits, dim=-1)
		valid_mask = (tgt != pad_id)

		correct_top1 = (pred_ids == tgt) & valid_mask
		top1_acc = correct_top1.sum().float() / valid_mask.sum().float()

		_, top5_ids = torch.topk(logits.float(), k=5, dim=-1)
		tgt_expanded = tgt.unsqueeze(-1).expand_as(top5_ids)
		correct_top5 = (top5_ids == tgt_expanded).any(dim=-1) & valid_mask
		top5_acc = correct_top5.sum().float() / valid_mask.sum().float()

		return top1_acc, top5_acc

	def training_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits = self(src, decoder_input)
		loss = self._calc_loss(tgt, logits)

		self.log("train_loss", loss, prog_bar=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

		logits = self(src, decoder_input)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

		self.val_top1_acc.append(top1_acc)
		self.val_top5_acc.append(top5_acc)

		self.log("val_loss", loss, prog_bar=True, sync_dist=True)
		self.log("v_top1", top1_acc, prog_bar=True, sync_dist=True)
		self.log("v_top5", top5_acc, prog_bar=True, sync_dist=True)

		return loss

	def on_validation_epoch_end(self):
		pass


	def test_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

		logits = self(src, decoder_input)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

		self.val_top1_acc.append(top1_acc)
		self.val_top5_acc.append(top5_acc)

		self.log("test_loss", loss, prog_bar=True, sync_dist=True)
		self.log("t_top1", top1_acc, prog_bar=True, sync_dist=True)
		self.log("t_top5", top5_acc, prog_bar=True, sync_dist=True)

		generated_tokens = self.model.generate(
			src.to(self.device),
			self.max_length,
			self.tokenizer.bos_token_id,
			self.tokenizer.eos_token_id,
		)
		generated_tokens = generated_tokens.cpu()

		gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
		# print("--------------------------")
		# print(gen_smiles_list)
		# print(ref_smiles_list)

		smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
		smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
		self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

		self.smiles_metric.update(gen_smiles_list, ref_smiles_list)
		torch.cuda.empty_cache()

		return {"test_loss": loss}

	def on_test_epoch_end(self):
		scores = self.smiles_metric.compute()

		self.log_dict({
			"t_valid": scores["valid_smiles_ratio"],
			"t_tanimoto": scores["avg_tanimoto"],
			"t_unique": scores["unique_ratio"],
			"t_dup_ratio": scores["duplicate_ratio"],
		}, prog_bar=True, sync_dist=True)

		self.smiles_metric.reset()

	def configure_optimizers(self):
		if self.mode == "pretrain":
			optimizer = torch.optim.AdamW(self.parameters(), lr=1.0, betas=(0.9, 0.999))
			d_model = self.model.d_model
			warmup_steps = 8000
			lr_lambda = lambda step: (d_model ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** -1.5))
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
			return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
		else:
			optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.999))
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max=10,
				eta_min=0
			)
			return [optimizer], [scheduler]
