import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import lr_scheduler

import lightning.pytorch as pl
from rdkit import Chem

from einops import rearrange

from typing import Optional, Tuple

from models.bart import BART
from models.sampler import greedy_sampler, beam_search_sampler, nucleus_sampler
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from metrics import SMILESEvaluationMetric, compute_batch_tanimoto_rewards

from utils import is_valid_smiles

import sys
from pprint import pprint
import time
import math
from models.lora import apply_lora_to_model

from lion_pytorch import Lion

class BARTModel(pl.LightningModule): 
	def __init__(
			self, model: BART, tokenizer: SMILESTokenizer | ChemformerTokenizer,
			mode: str = "pretrain",
			sampler: str = "greedy",
			kv_cache: bool = False,
			beam_size: int = 20,
			# aux_warmup_epochs: int = 10,
			aux_warmup_steps: int = 10000,
			aux_weight_max: float = 0.1,
			warm_up_percent: float = 0.05,
			rl_coef: float = 0.5,
			# rl_coef: float = 0,
			# mrt_coef: float = 0.5,
			mrt_coef: float = 0,
			mrt_beam_size: int = 10,
			mrt_alpha: float = 0.05,
			use_lora: bool = False,
		):
		super().__init__()
		# self.model = torch.compile(
		# 	model,
		# 	fullgraph=True,
		# )

		if use_lora:
			apply_lora_to_model(model, rank=16, alpha=8)

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

		self.mrt_coef = mrt_coef
		self.mrt_beam_size = mrt_beam_size
		self.mrt_alpha = mrt_alpha

		self.warm_up_percent = warm_up_percent
		self.total_steps = None
		self.warmup_steps = None

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

		generated_tokens, log_pi = None, None
		if self.rl_coef > 0:
			step = float(self.global_step)
			alpha = min(1.0, step / self.warmup_steps) if self.warmup_steps is not None else 0
			self.rl_weight = alpha * self.rl_coef

			idx = torch.randint(0, src.size(0), (1,)).item()

			bsz = src.size(0)
			# k = min(16, bsz)
			k = bsz
			# src_i = src[idx:idx+1] 
			# tgt_i = tgt[idx:idx+1]
			# mask_i = src_padding_mask[idx:idx+1]
			idx = torch.randperm(bsz)[:k]
			src_i = src[idx] 
			tgt_i = tgt[idx]
			mask_i = src_padding_mask[idx]

			with torch.no_grad():
				generated_tokens, log_pi = self.model.generate(
					src_i, mask_i, beam_search_sampler,
					max_length=self.max_length,
					start_token_id=self.tokenizer.bos_token_id,
					end_token_id=self.tokenizer.eos_token_id,
					beam_size=self.mrt_beam_size,
					length_penalty_alpha=0,
				) 

				# memory = self.model.encode(src_i, mask_i)

				# generated_tokens, _ = beam_search_sampler(
				# 	self.model, memory, mask_i,
				# 	start_token_id=self.tokenizer.bos_token_id,
				# 	end_token_id=self.tokenizer.eos_token_id,
				# 	beam_size=10,
				# ) 

				# generated_tokens = generated_tokens[:, 0, :]
				# log_pi, _, _ = self.model.evaluate_actions(
				# 	memory, mask_i, generated_tokens, self.tokenizer.pad_token_id
				# )
				_generated_tokens = generated_tokens[:, 0, :]
				_log_pi = log_pi[:, 0]

			# ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)
			ref_smiles_list = self.tokenizer.batch_decode(tgt_i, skip_special_tokens=True)
			gen_smiles_list = self.tokenizer.batch_decode(_generated_tokens, skip_special_tokens=True)

			self.smiles_metric.update(gen_smiles_list, ref_smiles_list)

			metrics = self.smiles_metric.compute_once(gen_smiles_list, ref_smiles_list)
			tanimoto, valid, exact, total = metrics["avg_tanimoto"], metrics["valid_smiles_ratio"], metrics["exact_match_ratio"], metrics["total_count"]

			valid_bonus = (valid * total * 0.5 + (1 - valid) * total * -1) / total

			reward = tanimoto
			# reward = tanimoto + 0.3 * exact + valid_bonus

			reward = torch.tensor(reward)

			self.baseline = 0.9 * self.baseline + 0.1 * reward.item()
			raw_rl = -(reward - self.baseline) * _log_pi.mean()
			rl_loss = raw_rl.clamp(min=-1, max=1)

			reward = reward.to(src.device)
			rl_loss = rl_loss.to(src.device)

			self.log("rl_loss", rl_loss, prog_bar=True, sync_dist=True)
			self.log("reward", reward, prog_bar=True, sync_dist=True)

			loss = loss + self.rl_weight * rl_loss

		if self.mrt_coef > 0:
			# Linearly warm up the MRT loss weight
			step = float(self.global_step)
			alpha = min(1.0, step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
			self.mrt_weight = alpha * self.mrt_coef

			with torch.no_grad():
				if generated_tokens is None:
					# beam scores without norm is raw probabilities
					generated_tokens, log_pi = self.model.generate(
						src, src_padding_mask, beam_search_sampler,
						max_length=self.max_length,
						start_token_id=self.tokenizer.bos_token_id,
						end_token_id=self.tokenizer.eos_token_id,
						beam_size=self.mrt_beam_size,
						length_penalty_alpha=0,
					)

			bsz, k, seq_len = generated_tokens.shape

			top_n = 3
			k = min(top_n, k)
			log_pi, top_indices = torch.topk(log_pi, k=k, dim=-1)

			indices_to_gather = top_indices.unsqueeze(-1).expand(bsz, k, seq_len)
			generated_tokens = torch.gather(generated_tokens, 1, indices_to_gather)
			
			ref_smiles_list = self.tokenizer.batch_decode(tgt, skip_special_tokens=True)

			flat_generated_tokens = rearrange(generated_tokens, 'b k s -> (b k) s')
			flat_generated_smiles = self.tokenizer.batch_decode(flat_generated_tokens, skip_special_tokens=True)
			repeated_ref_smiles = [smi for smi in ref_smiles_list for _ in range(k)]
			
			flat_rewards = compute_batch_tanimoto_rewards(flat_generated_smiles, repeated_ref_smiles)
			rewards = torch.tensor(flat_rewards, device=self.device, dtype=torch.float32).view(bsz, k)

			sharpened_log_pi = log_pi * self.mrt_alpha
			beam_probs = F.softmax(sharpened_log_pi, dim=-1)
			
			expected_reward = (beam_probs.detach() * rewards).sum(dim=-1)
			mrt_loss = -(beam_probs * rewards).sum(dim=-1).mean()
			
			loss = (1 - self.mrt_weight) * loss + self.mrt_weight * mrt_loss
			
			self.log("mrt_exp_reward", expected_reward.mean(), prog_bar=True, sync_dist=True)
			self.log("mrt_loss", mrt_loss, prog_bar=True, sync_dist=True)
			self.log("mrt_weight", self.mrt_weight, prog_bar=False, sync_dist=True)

		if aux_preds or self.rl_coef or self.mrt_coef > 0:
			self.log("total_loss", loss, prog_bar=True, sync_dist=True)

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
		# pprint(gen_smiles_list, stream=sys.stderr)
		# pprint(ref_smiles_list, stream=sys.stderr)
		# print("----------------------------------------", file=sys.stderr)
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
			# pprint(gen_smiles_list, stream=sys.stderr)
			# pprint(ref_smiles_list, stream=sys.stderr)
			# print("----------------------------------------", file=sys.stderr)
			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

			# print(ref_smiles_list)
			# print(gen_smiles_list)
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

			for i, gen_smile in enumerate(gen_smiles_list):
				gen_mol = Chem.MolFromSmiles(gen_smile)
				if gen_mol:
					gen_smiles_list[i] = Chem.MolToSmiles(gen_mol, canonical=True)

			smiles_correct = sum(1 for gen, ref in zip(gen_smiles_list, ref_smiles_list) if gen == ref)
			smiles_accuracy = smiles_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log("t_smi_top1", smiles_accuracy, prog_bar=True, sync_dist=True)

			for top_n in range(2, 4):
				top_n_correct = 0
				for i, ref in enumerate(ref_smiles_list):
					top_n_beams = generated_tokens[i, :top_n, :].cpu()
					top_n_smiles = self.tokenizer.batch_decode(top_n_beams, skip_special_tokens=True)
					if ref in top_n_smiles:
						top_n_correct += 1
				top_n_accuracy = top_n_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
				self.log(f"t_smi_top{top_n}", top_n_accuracy, prog_bar=True, sync_dist=True)
			
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
			scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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

			# optim = AdamW([
			# 	{"params": main_params, "lr": 5e-4, "weight_decay": 0.01},
			# 	{"params": no_decay, "lr": 5e-4, "weight_decay": 0.0},
			# 	{"params": aux_params, "lr": 1e-3, "weight_decay": 0.01},
			# ], betas=(0.9, 0.999))

			optim = Lion([
				{"params": main_params, "lr": 5e-4, "weight_decay": 0.01},
				{"params": no_decay, "lr": 5e-4, "weight_decay": 0.0},
				{"params": aux_params, "lr": 1e-3, "weight_decay": 0.01},
			], betas=(0.9, 0.999))

			if self.total_steps is None:
				self.total_steps = self.trainer.estimated_stepping_batches
				self.warmup_steps = self.warm_up_percent * self.total_steps

			# sched = lr_scheduler.OneCycleLR(
			# 	optim,
			# 	max_lr=[1e-3, 1e-3, 5e-3],
			# 	total_steps=self.total_steps,
			# 	pct_start=0.01,
			# 	anneal_strategy="cos",
			# 	div_factor=1e2,
			# 	final_div_factor=1e3,
			# 	cycle_momentum=False,
			# )

			sched = get_cosine_schedule_with_warmup(
				optim,
				num_warmup_steps=int(self.total_steps * self.warm_up_percent),
				num_training_steps=self.total_steps
			)

			return [optim], [
				{"scheduler": sched, "interval": "step"},
			]
