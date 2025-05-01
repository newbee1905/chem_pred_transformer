import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from muon import Muon

import lightning.pytorch as pl

from transformers import PreTrainedTokenizerFast
from rdkit import Chem

import torchmetrics
from torchmetrics.text.bleu import BLEUScore

from einops import rearrange, repeat

from typing import Optional, Tuple

from models.bart import BART
from models.sampler import greedy_sampler, beam_search_sampler
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from metrics import SMILESEvaluationMetric

from utils import is_valid_smiles

import time
import math

class BARTModel(pl.LightningModule): 
	def __init__(
			self, model: BART, tokenizer: SMILESTokenizer | ChemformerTokenizer,
			mode: str = "pretrain",
			sampler: str = "greedy", kv_cache: bool = False,
			beam_size: int = 10,
		):
		super().__init__()
		self.model = model
		self.tokenizer = tokenizer
		self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

		self.smiles_metric = SMILESEvaluationMetric()
		self.max_length = model.max_seq_len

		self.mode = mode
		self.sampler = sampler
		self.kv_cache = kv_cache
		self.beam_size = beam_size

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
		src, src_padding_mask, tgt = batch["input_ids"], batch["attention_mask"], batch["labels"]
		tgt_padding_mask = batch.get("labels_attention_mask", src_padding_mask)

		src_padding_mask = src_padding_mask.eq(0)
		tgt_padding_mask = tgt_padding_mask.eq(0)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)
		target = tgt[:, 1:]

		logits = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		self.log("train_loss", loss, prog_bar=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		src, src_padding_mask, tgt = batch["input_ids"], batch["attention_mask"], batch["labels"]
		tgt_padding_mask = batch.get("labels_attention_mask", src_padding_mask)

		src_padding_mask = src_padding_mask.eq(0)
		tgt_padding_mask = tgt_padding_mask.eq(0)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

		logits = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

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

		logits = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

		self.log("test_loss", loss, prog_bar=True, sync_dist=True)
		self.log("t_top1", top1_acc, prog_bar=True, sync_dist=True)
		self.log("t_top5", top5_acc, prog_bar=True, sync_dist=True)

		ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)

		if self.sampler == "greedy":
			generated_tokens = self.model.generate(
				src, None, greedy_sampler,
				max_length=self.max_length,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
				kv_cache=self.kv_cache,
			)
			generated_tokens = generated_tokens.cpu()
			gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

			print(ref_smiles_list)
			print(gen_smiles_list)
		else:
			generated_tokens, beam_scores = self.model.generate(
				src, None, beam_search_sampler,
				max_length=self.max_length,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
				beam_size=self.beam_size,
				# length_penalty_alpha=self.length_penalty_alpha,
				kv_cache=self.kv_cache,
			)

			top_beam_tokens = generated_tokens[:, 0, :].cpu()
			gen_smiles_list = self.tokenizer.batch_decode(top_beam_tokens, skip_special_tokens=True)

			print(gen_smiles_list)
			print(ref_smiles_list)
			print("----------------------------------------")

			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)
			
			if self.beam_size >= 5:
				top5_correct = 0
				for i, ref in enumerate(ref_smiles_list):
					top5_beams = generated_tokens[i, :5, :].cpu()
					top5_smiles = self.tokenizer.batch_decode(top5_beams, skip_special_tokens=True)
					if ref in top5_smiles:
						top5_correct += 1
				top5_accuracy = top5_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
				self.log("t_smi_top5", top5_accuracy, prog_bar=True, sync_dist=True)
			
			if self.beam_size >= 10:
				top10_correct = 0
				for i, ref in enumerate(ref_smiles_list):
					top10_beams = generated_tokens[i, :10, :].cpu()
					top10_smiles = self.tokenizer.batch_decode(top10_beams, skip_special_tokens=True)
					if ref in top10_smiles:
						top10_correct += 1
				top10_accuracy = top10_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
				self.log("t_smi_top10", top10_accuracy, prog_bar=True, sync_dist=True)
			
			self.log("t_beam_scores", beam_scores[:, 0].mean(), prog_bar=False, sync_dist=True)

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
			"t_dup_count": scores["duplicate_count"],
		}, prog_bar=True, sync_dist=True)

		self.smiles_metric.reset()

	def configure_optimizers(self):
		if self.mode == "pretrain":
			optimizer = AdamW(self.parameters(), lr=1.0, betas=(0.9, 0.999))
			d_model = self.model.d_model
			warmup_steps = 8000
			lr_lambda = lambda step: (d_model ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** -1.5))
			scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
			return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
		else:
			muon_params = [p for p in self.parameters() if p.ndim >= 2]
			adamw_params = [p for p in self.parameters() if p.ndim < 2]

			muon_optim = Muon(
				params=muon_params,
				lr=5e-5,
				momentum=0.95,
				nesterov=True,
				ns_steps=5,
				weight_decay=0.01,
				rank=self.global_rank,
				world_size=self.trainer.world_size,
			)
			adamw_optim = AdamW(adamw_params, lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)

			# scheduler = OneCycleLR(
			# 	optimizer,
			# 	max_lr=5e-5,
			# 	total_steps=total_steps,
			# 	pct_start=0.1,
			# 	anneal_strategy="cos",
			# 	div_factor=25.0,
			# 	final_div_factor=1e4,
			# )

			muon_sched = CosineAnnealingLR(
				muon_optim,
				T_max=self.trainer.max_epochs or 1,
				eta_min=1e-6,
			)

			adamw_sched = CosineAnnealingLR(
				adamw_optim,
				T_max=self.trainer.max_epochs or 1,
				eta_min=1e-6,
			)

			return [muon_optim, adamw_optim], [
				{"scheduler": muon_sched,  "interval": "step", "opt_idx": 0},
				{"scheduler": adamw_sched, "interval": "step", "opt_idx": 1},
			]
