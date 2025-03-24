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

from typing import Optional

from models.bart import BART
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from metrics import SMILESEvaluationMetric

class BARTModel(pl.LightningModule):
	def __init__(self, model: BART, tokenizer: SMILESTokenizer | ChemformerTokenizer, max_length: int = 256):
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

		self.generate_times = []

	def forward(self, src, tgt, src_mask, tgt_mask):
		return self.model(src, tgt)

	def training_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)
		attn_mask = mask.to(dtype=torch.float)
		attn_mask = mask.masked_fill(mask == 0, float('-inf'))
		attn_mask = mask.masked_fill(mask == 1, float(0.0))

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits = self(src, decoder_input, attn_mask, attn_mask)
		loss = self.loss_fn(
			logits[:, 1:, :].contiguous().view(-1, logits.size(-1)),
			target.contiguous().view(-1)
		)
		self.log("train_loss", loss, prog_bar=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits = self(src, decoder_input, mask, mask)
		loss = self.loss_fn(
			logits[:, 1:, :].contiguous().view(-1, logits.size(-1)),
			target.contiguous().view(-1)
		)

		mask_tokens = target != self.tokenizer.pad_token_id

		top1_preds = torch.argmax(logits[:, 1:, :], dim=-1)
		correct_top1 = (top1_preds == target) & mask_tokens
		top1_acc = correct_top1.sum().float() / mask_tokens.sum().float()

		_, top5_preds = torch.topk(logits[:, 1:, :], k=5, dim=-1)
		target_expanded = target.unsqueeze(-1).expand_as(top5_preds)
		correct_top5 = torch.any(top5_preds == target_expanded, dim=-1) & mask_tokens
		top5_acc = correct_top5.sum().float() / mask_tokens.sum().float()

		self.val_top1_acc.append(top1_acc)
		self.val_top5_acc.append(top5_acc)
		self.log("val_loss", loss, prog_bar=True, sync_dist=True)

		if self.current_epoch % 5 == 4:
			start_time = time.time()
			generated_tokens = self.model.generate(
				src.to(self.device),
				self.max_length,
				self.tokenizer.bos_token_id,
				self.tokenizer.eos_token_id,
			)
			generated_tokens = generated_tokens.cpu()
			self.generate_times.append(time.time() - start_time)

			gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
			# print(gen_smiles_list, ref_smiles_list)

			self.smiles_metric.update(gen_smiles_list, ref_smiles_list)

			torch.cuda.empty_cache()

		return {"val_loss": loss}

	def on_validation_epoch_end(self):
		avg_top1 = torch.stack(self.val_top1_acc).mean()
		avg_top5 = torch.stack(self.val_top5_acc).mean()

		if self.current_epoch % 5 == 4:
			scores = self.smiles_metric.compute()

			avg_gen_ct = (sum(self.generate_times) / len(self.generate_times) if self.generate_times else 0.0)

			self.log_dict({
				"val_top1_acc": avg_top1,
				"val_top5_acc": avg_top5,
				"val_valid_smiles_ratio": scores["valid_smiles_ratio"],
				"val_avg_tanimoto": scores["avg_tanimoto"],
				"val_gen_ct": avg_gen_ct,
			}, prog_bar=True, sync_dist=True)
		else:
			self.log_dict({
				"val_top1_acc": avg_top1,
				"val_top5_acc": avg_top5,
			}, prog_bar=True, sync_dist=True)

		self.val_top1_acc.clear()
		self.val_top5_acc.clear()

		self.generate_times.clear()
		self.smiles_metric.reset()

	def test_step(self, batch, batch_idx):
		src, tgt = batch["input_ids"], batch["labels"]
		mask = batch["attention_mask"].to(torch.float)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits = self(src, decoder_input, mask, mask)
		loss = self.loss_fn(
			logits[:, 1:, :].contiguous().view(-1, logits.size(-1)),
			target.contiguous().view(-1)
		)

		mask_tokens = target != self.tokenizer.pad_token_id

		top1_preds = torch.argmax(logits[:, 1:, :], dim=-1)
		correct_top1 = (top1_preds == target) & mask_tokens
		top1_acc = correct_top1.sum().float() / mask_tokens.sum().float()

		_, top5_preds = torch.topk(logits[:, 1:, :], k=5, dim=-1)
		target_expanded = target.unsqueeze(-1).expand_as(top5_preds)
		correct_top5 = torch.any(top5_preds == target_expanded, dim=-1) & mask_tokens
		top5_acc = correct_top5.sum().float() / mask_tokens.sum().float()

		self.test_top1_acc.append(top1_acc)
		self.test_top5_acc.append(top5_acc)
		self.log("test_loss", loss, prog_bar=True, sync_dist=True)

		start_time = time.time()
		generated_tokens = self.model.generate(
			src.to(self.device),
			self.max_length,
			self.tokenizer.bos_token_id,
			self.tokenizer.eos_token_id,
		)
		generated_tokens = generated_tokens.cpu()
		self.generate_times.append(time.time() - start_time)

		gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
		print("-------------------------")
		print(gen_smiles_list)
		print(ref_smiles_list)

		self.smiles_metric.update(gen_smiles_list, ref_smiles_list)
		torch.cuda.empty_cache()

		return {"test_loss": loss}

	def on_test_epoch_end(self):
		avg_top1 = torch.stack(self.test_top1_acc).mean()
		avg_top5 = torch.stack(self.test_top5_acc).mean()

		scores = self.smiles_metric.compute()

		avg_gen_ct = (sum(self.generate_times) / len(self.generate_times) if self.generate_times else 0.0)

		self.log_dict({
			"test_top1_acc": avg_top1,
			"test_top5_acc": avg_top5,
			"test_valid_smiles_ratio": scores["valid_smiles_ratio"],
			"test_avg_tanimoto": scores["avg_tanimoto"],
			"test_gen_ct": avg_gen_ct,
		}, prog_bar=True, sync_dist=True)

		self.test_top1_acc.clear()
		self.test_top5_acc.clear()

		self.generate_times.clear()
		self.smiles_metric.reset()

	def configure_optimizers(self):
		return torch.optim.AdamW(self.parameters(), lr=5e-5)
