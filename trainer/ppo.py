import copy
import re

from typing import Optional, Callable, Dict, Any, Tuple

from models.bart import BART
from models.chemformer import Chemformer
from rdkit import Chem

from metrics import SMILESEvaluationMetric, compute_batch_tanimoto_rewards
from models.sampler import greedy_sampler, beam_search_sampler, nucleus_sampler
from models.rl import compute_intermediate_rewards, Critic
from transformers import get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

LOG_RATIO_CLAMP_RANGE = (-5.0, 5.0)
GRAD_CLIP_NORM = 0.5

class PPOModule(pl.LightningModule):
	def __init__(
		self,
		actor: BART | Chemformer,
		critic: Critic,
		tokenizer,
		sampler_fn: Callable = beam_search_sampler,
		sampler_kwargs: Optional[Dict[str, Any]] = None,
		lr: float = 5e-5,
		ppo_epochs: int = 5,
		clip_epsilon: float = 0.2,
		ent_coef: float = 0.05,
		kl_coef: float = 0.1,
		warm_up_percent: float = 0.1,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		is_per_step: bool = False,
	):
		super().__init__()
		self.save_hyperparameters(ignore=["actor", "critic", "tokenizer"])

		self.actor = actor
		self.critic = critic
		self.critic.is_per_step = is_per_step
		self.tokenizer = tokenizer
		self.sampler_fn = sampler_fn

		self.total_steps = None

		self.ref_actor = self._create_reference_actor()
		self.sampler_kwargs = self._configure_sampler_kwargs(sampler_kwargs)

		self.automatic_optimization = False
		self.smiles_metric = SMILESEvaluationMetric()

	def _create_reference_actor(self):
		"""Creates a deep copy of the actor to use as a frozen reference."""

		ref_actor = copy.deepcopy(self.actor)

		for param in ref_actor.parameters():
			param.requires_grad = False

		ref_actor.eval()

		return ref_actor

	def _configure_sampler_kwargs(self, user_kwargs: Optional[Dict]) -> Dict:
		"""Sets up default arguments for the chosen sampler function."""

		kwargs = user_kwargs or {}

		kwargs.setdefault("start_token_id", self.tokenizer.bos_token_id)
		kwargs.setdefault("end_token_id", self.tokenizer.eos_token_id)

		if self.sampler_fn == nucleus_sampler:
			kwargs.setdefault("top_p", 0.9)
		elif self.sampler_fn == beam_search_sampler:
			kwargs.setdefault("beam_size", 10)

		return kwargs

	def _rollout_experience(
		self, src_tokens: torch.Tensor, src_mask: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Generate experience from the current policy."""
		with torch.no_grad():
			memory = self.actor.encode(src_tokens, src_mask)
			self.actor.clear_cache()

			if self.sampler_fn == beam_search_sampler:
				generated_tokens, _ = self.sampler_fn(
					self.actor, memory, src_mask, kv_cache=False, **self.sampler_kwargs
				)
				pred_tokens = generated_tokens[:, 0, :]
			else:
				pred_tokens = self.sampler_fn(
					self.actor, memory, src_mask, kv_cache=True, **self.sampler_kwargs
				)
			
			self.actor.clear_cache()

			old_log_probs, _, _ = self.actor.evaluate_actions(
				memory, src_mask, pred_tokens, self.tokenizer.pad_token_id
			)

			decoder_input_for_value = pred_tokens[:, :-1]
			decoder_output_for_value = self.actor.decode(
				decoder_input_for_value, memory, memory_mask=src_mask
			)
			values = self.critic(decoder_output_for_value, memory, src_mask)

		return pred_tokens, old_log_probs, values, memory
	
	def _compute_rewards(self, pred_tokens: torch.Tensor, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
		"""Compute rewards based on Tanimoto similarity and SMILES grammar."""
		pred_smiles = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
		reactant_smiles = self.tokenizer.batch_decode(src_tokens, skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens, skip_special_tokens=True)

		tanimoto_rewards = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)

		bsz, seq_len = pred_tokens.shape
		all_prefixes = [
			self.tokenizer.decode(pred_tokens[b, : i + 1], skip_special_tokens=True)
			for b in range(bsz) for i in range(seq_len)
		]
		all_reactants = [reactant_smiles[b] for b in range(bsz) for _ in range(seq_len)]
		
		grammar_scores_flat = compute_intermediate_rewards(all_prefixes, all_reactants)
		grammar_scores = torch.tensor(grammar_scores_flat, device=self.device).view(bsz, seq_len)
		grammar_rewards = grammar_scores.mean(dim=1)
		
		return tanimoto_rewards + grammar_rewards

	def _compute_per_step_rewards(self, pred_tokens: torch.Tensor, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
		"""Compute rewards based on Tanimoto similarity and SMILES grammar."""
		pred_smiles = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
		reactant_smiles = self.tokenizer.batch_decode(src_tokens, skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens, skip_special_tokens=True)

		tanimoto_rewards = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)

		bsz, seq_len = pred_tokens.shape
		all_prefixes = [
			self.tokenizer.decode(pred_tokens[b, : i + 1], skip_special_tokens=True)
			for b in range(bsz) for i in range(seq_len)
		]
		all_reactants = [reactant_smiles[b] for b in range(bsz) for _ in range(seq_len)]
		
		grammar_scores_flat = compute_intermediate_rewards(all_prefixes, all_reactants)
		grammar_scores = torch.tensor(grammar_scores_flat, device=self.device).view(bsz, seq_len)

		step_rewards = torch.diff(grammar_scores, dim=1, prepend=torch.zeros((bsz, 1), device=self.device))
		tanimoto_rewards = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)
		action_mask = pred_tokens.ne(self.tokenizer.pad_token_id)
		sequence_lengths = action_mask.sum(dim=1)

		step_rewards[torch.arange(bsz), sequence_lengths - 1] += tanimoto_rewards
		
		return step_rewards

	def _compute_gae(
		self, rewards: torch.Tensor, values: torch.Tensor, action_mask: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute Generalized Advantage Estimation."""
		num_steps = values.size(1)
		advantages = torch.zeros_like(values)
		last_gae_lam = 0

		rewards_trimmed = rewards[:, :num_steps]
		mask_trimmed = action_mask[:, :num_steps]

		# Append a zero value for the terminal state
		values_extended = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)

		for t in reversed(range(num_steps)):
			mask = action_mask[:, t].float()

			# TD Error: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
			delta = rewards[:, t] + self.hparams.gamma * values_extended[:, t] * mask - values_extended[:, t]

			# GAE: A_t = delta_t + gamma * lambda * A_{t+1}
			last_gae_lam = delta + self.hparams.gamma * self.hparams.gae_lambda * last_gae_lam * mask
			advantages[:, t] = last_gae_lam

		returns = advantages + values
		return advantages, returns

	def training_step(self, batch: dict, batch_idx: int):
		opt_actor, opt_critic = self.optimizers()

		self.actor.eval()
		self.critic.eval()

		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask")
		src_mask = src_mask.eq(0)
		tgt_tokens = batch["labels"]

		pred_tokens, old_log_probs, values, memory = self._rollout_experience(src_tokens, src_mask)
		
		if not self.hparams.is_per_step:
			rewards = self._compute_rewards(pred_tokens, src_tokens, tgt_tokens)
			returns = rewards.detach()
			adv = (returns - values).detach()

			# Normalize advantages for training stability 
			adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
		else:
			action_mask = pred_tokens.ne(self.tokenizer.pad_token_id)
			per_step_rewards = self._compute_per_step_rewards(pred_tokens, src_tokens, tgt_tokens)
			advs, returns = self._compute_gae(per_step_rewards, values, action_mask)

			# adv = advs[action_mask]
			adv = advs[action_mask[:, :advs.shape[1]]]
			adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

		self.actor.train()
		self.critic.train()

		actions = pred_tokens.detach()
		for _ in range(self.hparams.ppo_epochs):
			self.actor.clear_cache()

			new_log_probs, entropy, decoder_output = self.actor.evaluate_actions(
				memory, src_mask, pred_tokens, self.tokenizer.pad_token_id
			)

			with torch.no_grad():
				self.actor.clear_cache()

				ref_log_probs, _, _ = self.ref_actor.evaluate_actions(
					memory.detach(), src_mask, pred_tokens, self.tokenizer.pad_token_id
				)

			ref_log_probs = ref_log_probs.to(self.device)
			new_log_probs = new_log_probs.to(old_log_probs.device)

			# Policy Loss (Clipped Surrogate Objective)
			log_ratio = torch.clamp(new_log_probs - old_log_probs.detach(), *LOG_RATIO_CLAMP_RANGE)
			ratio = log_ratio.exp().to(adv.device)

			if not self.hparams.is_per_step:
				s1 = ratio * adv
				s2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * adv
			else:
				active_mask = action_mask[:, :ratio.shape[1]]
				s1 = ratio[active_mask] * adv
				s2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * adv

			policy_loss = -torch.min(s1, s2).mean()

			# Entropy Bonus & KL Penalty
			entropy_bonus = self.hparams.ent_coef * entropy.mean()
			# kl_div = F.kl_div(ref_log_probs, new_log_probs, log_target=True, reduction='batchmean')
			kl_div = F.kl_div(new_log_probs, ref_log_probs, log_target=True, reduction='batchmean')
			kl_penalty = self.hparams.kl_coef * kl_div

			actor_loss = policy_loss - entropy_bonus + kl_penalty

			# Value Function Loss
			new_values = self.critic(decoder_output.detach(), memory.detach(), src_mask)
			if not self.hparams.is_per_step:
				value_loss = F.mse_loss(new_values, returns)
			else:
				log_probs_flat = new_log_probs[action_mask]
				old_log_probs_flat = old_log_probs[action_mask]

				values_flat = new_values[action_mask]
				returns_flat = returns[action_mask]

				value_loss = F.mse_loss(values_flat, returns_flat)

			# --- Optimization Step ---
			opt_actor.zero_grad()
			self.manual_backward(actor_loss)
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
			opt_actor.step()

			opt_critic.zero_grad()
			self.manual_backward(value_loss)
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)
			opt_critic.step()

		print(f"\n--- DEBUGGING at Epoch {self.current_epoch}, Batch Index {batch_idx} ---")
		print(f"Rewards:         mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
		print(f"Values (critic): mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
		print(f"Returns:         mean={returns.mean():.4f}, std={returns.std():.4f}, min={returns.min():.4f}, max={returns.max():.4f}")

		self.log_dict({
			"train/policy_loss": policy_loss,
			'train/actor_loss':  actor_loss,
			"train/value_loss": value_loss,
			"train/mean_reward": rewards.mean(),
			"train/entropy": entropy.mean().item(),
			"train/kl_penalty": kl_penalty.item(),
			"train/ratio_mean": ratio.mean().item()
		}, prog_bar=True, on_step=True, on_epoch=False)

	def validation_step(self, batch: dict, batch_idx: int):
		self.actor.eval()
		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask")
		src_mask = src_mask.eq(0)
		tgt_tokens = batch["labels"]

		beam_size = 10 
		max_length = self.actor.max_seq_len

		with torch.no_grad():
			# pred_tokens = self.actor.generate(
			# 	src_tokens,
			# 	src_mask,
			# 	sampler=greedy_sampler,
			# 	kv_cache=True,
			# 	start_token_id=self.tokenizer.bos_token_id,
			# 	end_token_id=self.tokenizer.eos_token_id,
			# )
			generated_tokens, beam_scores = self.actor.generate(
				src_tokens,
				src_mask,
				sampler=beam_search_sampler,
				beam_size=beam_size,
				max_length=max_length,
				kv_cache=False,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
			)
			pred_tokens = generated_tokens[:, 0, :]

		pred_smiles = self.tokenizer.batch_decode(pred_tokens.tolist(), skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)

		for i, pred_smile in enumerate(pred_smiles):
			pred_mol = Chem.MolFromSmiles(pred_smile)
			if pred_mol:
				pred_smiles[i] = Chem.MolToSmiles(pred_mol, canonical=True)
		
		self.smiles_metric.update(pred_smiles, target_smiles)

		rewards = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)

		batch_size = len(target_smiles)
		exact_match_count = sum(p == t for p, t in zip(pred_smiles, target_smiles))
		exact_match_rate = exact_match_count / batch_size

		self.log_dict({
			"val/mean_reward": rewards.mean(),
			"val/exact_match_rate": exact_match_rate
		}, prog_bar=True, on_epoch=True, sync_dist=True)

		return rewards.mean()

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

	def test_step(self, batch: dict, batch_idx: int):
		self.actor.eval()
		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask")
		src_mask = src_mask.eq(0)
		tgt_tokens = batch["labels"]

		beam_size = 10 
		max_length = self.actor.max_seq_len

		with torch.no_grad():
			generated_tokens, beam_scores = self.actor.generate(
				src_tokens,
				src_mask,
				sampler=beam_search_sampler,
				beam_size=beam_size,
				max_length=max_length,
				kv_cache=False,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
			)

		ref_smiles_list = self.tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)
		
		# --- Top-1 Metrics ---
		top_beam_tokens = generated_tokens[:, 0, :]
		top_1_smiles_list = self.tokenizer.batch_decode(top_beam_tokens.tolist(), skip_special_tokens=True)

		smiles_correct_top1 = sum(1 for gen, ref in zip(top_1_smiles_list, ref_smiles_list) if gen == ref)
		smiles_accuracy_top1 = smiles_correct_top1 / len(ref_smiles_list) if ref_smiles_list else 0.0
		self.log("test/smi_accuracy_top1", smiles_accuracy_top1, prog_bar=True, sync_dist=True)

		self.smiles_metric.update(top_1_smiles_list, ref_smiles_list)

		rewards = compute_batch_tanimoto_rewards(top_1_smiles_list, ref_smiles_list, device=self.device)
		self.log("test/tanimoto_top1", rewards.mean(), prog_bar=True, sync_dist=True)

		# --- Top-N Metrics (for N=5, 10) ---
		for top_n in range(5, beam_size + 1, 5):
			top_n_correct = 0
			for i, ref in enumerate(ref_smiles_list):
				top_n_beams = generated_tokens[i, :top_n, :]
				top_n_smiles = self.tokenizer.batch_decode(top_n_beams.tolist(), skip_special_tokens=True)

				if ref in top_n_smiles:
					top_n_correct += 1
			top_n_accuracy = top_n_correct / len(ref_smiles_list) if ref_smiles_list else 0.0
			self.log(f"test/smi_accuracy_top{top_n}", top_n_accuracy, prog_bar=True, sync_dist=True)
		
		self.log("test/beam_scores", beam_scores[:, 0].mean(), prog_bar=False, sync_dist=True)

		return rewards.mean()

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
		ff_params = []
		emb_params = []
		no_decay = []
		main_params = []

		for name, p in self.actor.named_parameters():
			if not p.requires_grad:
				continue
			if "embed_tokens" in name:
				emb_params.append(p)
			elif "ff" in name:
				ff_params.append(p)
			elif "bias" in name or "norm" in name:
				no_decay.append(p)
			else:
				main_params.append(p)

		base_lr = self.hparams.lr
		optimizer_grouped_parameters = [
			{"params": main_params, "weight_decay": 0.01},
			{"params": no_decay, "weight_decay": 0.0},
			{"params": ff_params, "lr": base_lr * 0.5, "weight_decay": 0.01},
			{"params": emb_params, "lr": base_lr * 0.1, "weight_decay": 0.0},
		]

		opt_actor = torch.optim.AdamW(optimizer_grouped_parameters, lr=base_lr, betas=(0.9, 0.999))
		opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=self.hparams.lr)

		if self.total_steps is None:
			self.total_steps = self.trainer.estimated_stepping_batches
			self.warmup_steps = int(self.hparams.warm_up_percent * self.total_steps)

		actor_scheduler = get_cosine_schedule_with_warmup(
			opt_actor,
			num_warmup_steps=self.warmup_steps,
			num_training_steps=self.total_steps
		)

		return (
			[opt_actor, opt_critic],
			[
				{"scheduler": actor_scheduler, "interval": "step"},
			]
		)
