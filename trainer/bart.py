import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim import lr_scheduler

import lightning.pytorch as pl

from einops import rearrange

from typing import Optional, Tuple

from models.bart import BART
from models.sampler import greedy_sampler, beam_search_sampler
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from metrics import SMILESEvaluationMetric

from utils import is_valid_smiles

import sys
from pprint import pprint
import time
import math

class BARTModel(pl.LightningModule): 
	def __init__(
			self, model: BART, tokenizer: SMILESTokenizer | ChemformerTokenizer,
			mode: str = "pretrain",
			sampler: str = "greedy", kv_cache: bool = False,
			beam_size: int = 20,
			# aux_warmup_epochs: int = 10,
			aux_warmup_steps: int = 10000,
			aux_weight_max: float = 0.1,
			rl_coef: float = 0.1,
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

		# self.aux_warmup_epochs = aux_warmup_epochs
		self.aux_warmup_steps = aux_warmup_steps
		self.aux_weight_max = aux_weight_max
		self.aux_weight = 0 

		self.aux_loss_fn = nn.GaussianNLLLoss(full=False, eps=1e-6)

		self.rl_coef = rl_coef
		self.baseline = 0.0

		# self.automatic_optimization = False


	def forward(self, src, tgt, src_mask = None, tgt_mask = None):
		out = self.model(src, tgt, src_mask, tgt_mask)
		if self.model.aux_head:
			logits, aux_preds = out
		else:
			logits, aux_preds = out, {}
		return logits, aux_preds

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

		logits, aux_preds = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		self.log("train_loss", loss, prog_bar=True, sync_dist=True)

		if aux_preds:
			step = float(self.global_step)
			warmup_steps = self.aux_warmup_steps * self.trainer.num_training_batches
			alpha = min(1.0, step / warmup_steps)
			self.aux_weight = alpha * self.aux_weight_max

			targets = torch.stack(
				[batch[f"aux_{name}"].to(pred) for name, pred in aux_preds.items()],
				dim=1,
			).to(self.device)

			preds = torch.stack(list(aux_preds.values()), dim=1)
			# expand to match batch_size and repeat for react and pred
			bsz, K = preds.size()
			logvars = torch.exp(self.model.aux_logvars)
			# logvars = logvars.repeat(2)
			logvars = logvars.unsqueeze(0).expand(bsz, K)

			aux_loss = self.aux_loss_fn(preds, targets, logvars)

			self.log("train_aux_loss", aux_loss, prog_bar=True, sync_dist=True)
			loss = loss + self.aux_weight * aux_loss
			# self.log("train_total_loss", loss, prog_bar=True, sync_dist=True)

		if self.rl_coef > 0:
			with torch.no_grad():
				generated_tokens, log_pi = self.model.generate(
					src, src_padding_mask, greedy_sampler,
					max_length=self.max_length,
					start_token_id=self.tokenizer.bos_token_id,
					end_token_id=self.tokenizer.eos_token_id,
					kv_cache=self.kv_cache,
					return_logpi=True,
				) 

			ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
			gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

			self.smiles_metric.update(gen_smiles_list, ref_smiles_list)

			metrics = self.smiles_metric.compute_once(gen_smiles_list, ref_smiles_list)
			reward = torch.tensor(metrics["avg_tanimoto"])

			self.baseline = 0.9 * self.baseline + 0.1 * reward.item()
			rl_loss = -(reward - self.baseline) * log_pi.mean()

			reward = reward.to(src.device)
			rl_loss = rl_loss.to(src.device)

			self.log("train_rl_loss", rl_loss, prog_bar=True, sync_dist=True)
			self.log("train_reward", reward, prog_bar=True, sync_dist=True)

			loss = loss + self.rl_coef * rl_loss

		if aux_preds or self.rl_coef > 0:
			self.log("train_total_loss", loss, prog_bar=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		src, src_padding_mask, tgt = batch["input_ids"], batch["attention_mask"], batch["labels"]
		tgt_padding_mask = batch.get("labels_attention_mask", src_padding_mask)

		src_padding_mask = src_padding_mask.eq(0)
		tgt_padding_mask = tgt_padding_mask.eq(0)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

		logits, aux_preds = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

		self.log("val_loss", loss, prog_bar=True, sync_dist=True)
		self.log("v_top1", top1_acc, prog_bar=True, sync_dist=True)
		self.log("v_top5", top5_acc, prog_bar=True, sync_dist=True)

		if aux_preds:
			targets = torch.stack(
				[batch[f"aux_{name}"].to(pred) for name, pred in aux_preds.items()],
				dim=1,
			).to(self.device)

			preds = torch.stack(list(aux_preds.values()), dim=1)
			# expand to match batch_size and repeat for react and pred
			bsz, K = preds.size()
			logvars = torch.exp(self.model.aux_logvars)
			# logvars = logvars.repeat(2)
			logvars = logvars.unsqueeze(0).expand(bsz, K)

			aux_loss = self.aux_loss_fn(preds, targets, logvars)

			self.log("v_aux_loss", aux_loss, prog_bar=True, sync_dist=True)
			loss = loss + self.aux_weight * aux_loss
			self.log("v_total_loss", loss, prog_bar=True, sync_dist=True)

		ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)

		generated_tokens = self.model.generate(
			src, src_padding_mask, greedy_sampler,
			max_length=self.max_length,
			start_token_id=self.tokenizer.bos_token_id,
			end_token_id=self.tokenizer.eos_token_id,
			kv_cache=self.kv_cache,
		)
		generated_tokens = generated_tokens.cpu()
		gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
		pprint(gen_smiles_list, stream=sys.stderr)
		pprint(ref_smiles_list, stream=sys.stderr)
		print("----------------------------------------", file=sys.stderr)
		smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
		smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
		self.log("v_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

		self.smiles_metric.update(gen_smiles_list, ref_smiles_list)
		torch.cuda.empty_cache()

		return loss

	def on_validation_epoch_end(self):
		scores = self.smiles_metric.compute()

		self.log_dict({
			"v_valid": scores["valid_smiles_ratio"],
			"v_tanimoto": scores["avg_tanimoto"],
			"v_unique": scores["unique_ratio"],
			"v_dup_ratio": scores["duplicate_ratio"],
			"v_dup_count": scores["duplicate_count"],
		}, prog_bar=True, sync_dist=True)

		self.smiles_metric.reset()

	def test_step(self, batch, batch_idx):
		src, src_padding_mask, tgt = batch["input_ids"], batch["attention_mask"], batch["labels"]
		tgt_padding_mask = batch.get("labels_attention_mask", src_padding_mask)

		src_padding_mask = src_padding_mask.eq(0)
		tgt_padding_mask = tgt_padding_mask.eq(0)

		bos = torch.full((tgt.size(0), 1), self.tokenizer.bos_token_id, device=self.device, dtype=torch.long)
		decoder_input = torch.cat([bos, tgt[:, :-1]], dim=1)

		logits, aux_preds = self(src, decoder_input, src_padding_mask, tgt_padding_mask)
		loss = self._calc_loss(tgt, logits)

		top1_acc, top5_acc = self._calc_token_acc(tgt, logits)

		self.log("test_loss", loss, prog_bar=True, sync_dist=True)
		self.log("t_top1", top1_acc, prog_bar=True, sync_dist=True)
		self.log("t_top5", top5_acc, prog_bar=True, sync_dist=True)

		if aux_preds:
			targets = torch.stack(
				[batch[f"aux_{name}"].to(pred) for name, pred in aux_preds.items()],
				dim=1,
			).to(self.device)

			preds = torch.stack(list(aux_preds.values()), dim=1)
			# expand to match batch_size and repeat for react and pred
			bsz, K = preds.size()
			logvars = torch.exp(self.model.aux_logvars)
			# logvars = logvars.repeat(2)
			logvars = logvars.unsqueeze(0).expand(bsz, K)

			aux_loss = self.aux_loss_fn(preds, targets, logvars)

			self.log("t_aux_loss", aux_loss, prog_bar=True, sync_dist=True)
			loss = loss + self.aux_weight * aux_loss
			self.log("t_total_loss", loss, prog_bar=True, sync_dist=True)

		ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)

		if self.sampler == "greedy":
			generated_tokens = self.model.generate(
				src, src_padding_mask, greedy_sampler,
				max_length=self.max_length,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
				kv_cache=self.kv_cache,
			)
			generated_tokens = generated_tokens.cpu()
			gen_smiles_list = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
			pprint(gen_smiles_list, stream=sys.stderr)
			pprint(ref_smiles_list, stream=sys.stderr)
			print("----------------------------------------", file=sys.stderr)
			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

			print(ref_smiles_list)
			print(gen_smiles_list)
		else:
			generated_tokens, beam_scores = self.model.generate(
				src, src_padding_mask, beam_search_sampler,
				max_length=self.max_length,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
				beam_size=self.beam_size,
				# length_penalty_alpha=self.length_penalty_alpha,
				kv_cache=self.kv_cache,
			)

			top_beam_tokens = generated_tokens[:, 0, :].cpu()
			gen_smiles_list = self.tokenizer.batch_decode(top_beam_tokens, skip_special_tokens=True)

			pprint(gen_smiles_list, stream=sys.stderr)
			pprint(ref_smiles_list, stream=sys.stderr)
			print("----------------------------------------", file=sys.stderr)

			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)
			
			for top_n in range(5, self.beam_size + 1, 5):
				top_n_correct = 0
				for i, ref in enumerate(ref_smiles_list):
					top_n_beams = generated_tokens[i, :top_n, :].cpu()
					top_n_smiles = self.tokenizer.batch_decode(top_n_beams, skip_special_tokens=True)
					if ref in top_n_smiles:
						top_n_correct += 1
				top_n_accuracy = top_n_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
				self.log(f"t_smi_top{top_n}", top_n_accuracy, prog_bar=True, sync_dist=True)
			
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
			lr_lambda = lambda step: (d_model ** -0.5) * min(
				(step + 1) ** (-0.5),
				(step + 1) * (warmup_steps ** -1.5)
			)
			scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
			return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
		else:
			main_params = []
			aux_params = []
			no_decay = []
			for name, p in self.named_parameters():
				if "aux" in name or "shared_proj" in name:
					aux_params.append(p)
				elif "bias" in name or "norm" in name:
					no_decay.append(p)
				else:
					main_params.append(p)

			optim = AdamW([
				{"params": main_params, "lr": 1e-3, "weight_decay": 0.01},
				{"params": no_decay, "lr": 1e-3, "weight_decay": 0.0},
				{"params": aux_params, "lr": 5e-3, "weight_decay": 0.01},
			], betas=(0.9, 0.999))


			total_steps = self.trainer.estimated_stepping_batches
			sched = lr_scheduler.OneCycleLR(
				optim,
				max_lr=[1e-3, 1e-3, 5e-3],
				total_steps=total_steps,
				pct_start=0.1,
				anneal_strategy="cos",
				div_factor=1e2,
				final_div_factor=1e3,
				cycle_momentum=False,
			)
			# sched = get_linear_schedule_with_warmup(
			# 	optim,
			# 	num_warmup_steps=int(total_steps * 0.1),
			# 	num_training_steps=total_steps
			# )

			return [optim], [
				{"scheduler": sched, "interval": "step"},
			]
