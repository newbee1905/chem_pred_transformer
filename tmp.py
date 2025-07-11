import argparse
import copy
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import re

from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from tokenisers.neochem import ChemformerTokenizerFast
from models.mha import KVCacheMHA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset.zinc import load_smiles_by_set
from dataset.uspto import USPTODataset
from dataset.base import BARTDataCollator

import importlib
import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from utils import set_seed, filter_none_kwargs
from concurrent.futures import ThreadPoolExecutor

import torch._dynamo
from torch._dynamo import disable

torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.cache_size_limit = 256
# pl.LightningModule.log = disable(pl.LightningModule.log)

from trainer.bart import BARTModel
from models.bart import BART
from metrics import compute_batch_tanimoto_rewards
from models.chemformer import Chemformer
from models.sampler import greedy_sampler, beam_search_sampler, nucleus_sampler
from transformers import get_cosine_schedule_with_warmup

from typing import Optional, Callable, Dict, Any

class LoRAInjectedLinear(nn.Module):
	"""
	A wrapper class that injects a LoRA layer into an existing nn.Linear layer.
	"""
	def __init__(self, original_layer: nn.Linear, rank: int, alpha: int = 8):
		super().__init__()
		
		self.original_layer = original_layer
		self.original_layer.requires_grad_(False)
		
		in_features, out_features = original_layer.in_features, original_layer.out_features
		self.in_features = in_features
		self.out_features = out_features
		self.rank = rank
		self.alpha = alpha

		self.lora_A = nn.Linear(in_features, rank, bias=False)
		self.lora_B = nn.Linear(rank, out_features, bias=False)
		
		# Init LoRA weights
		# A is initialized with Kaiming uniform, B is initialized to zero.
		# This ensures that at the start of training, the LoRA modification is zero.
		nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
		nn.init.zeros_(self.lora_B.weight)
		
		self.scaling = self.alpha / self.rank

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		original_output = self.original_layer(x)
		lora_output = self.lora_B(self.lora_A(x)) * self.scaling
		
		return original_output + lora_output

def apply_lora_to_model(model: nn.Module, rank: int, alpha: int):
	for name, module in model.named_children():
		if isinstance(module, nn.Linear):
			setattr(model, name, LoRAInjectedLinear(module, rank, alpha))
		else:
			apply_lora_to_model(module, rank, alpha)

USPTO_CSV_FILE = "USPTO_MIT.csv"
MAX_LENGTH = 282
BATCH_SIZE = 256
NUM_WORKERS = 24
NUM_EPOCHS = 10
# CKPT_PATH = "train_checkpoints_server/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
CKPT_PATH = "train_checkpoints/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"

VALID_SMILES_CHARS_RE = re.compile(r'[^\w\(\)\[\]=\-\#/\+\.@%0-9]')
INVALID_BOND_SEQUENCE_RE = re.compile(r'==|##|==#|##=')
ATOM_PATTERN_RE = re.compile(r'[A-Z][a-z]?')

def _intermediate_smiles_reward(
	partial_smiles: str,
	reactant_smiles: str
) -> float:
	"""
	Performs syntax checks and atom-count matching for a partial SMILES.
	"""
	# Syntax heuristics
	if VALID_SMILES_CHARS_RE.search(partial_smiles):
		return 0.0
	if (partial_smiles.count('(') < partial_smiles.count(')') or
			partial_smiles.count('[') < partial_smiles.count(']')):
		return 0.0
	if INVALID_BOND_SEQUENCE_RE.search(partial_smiles):
		return 0.0

	# Atom counts
	reac_atoms = ATOM_PATTERN_RE.findall(reactant_smiles)
	if not reac_atoms:
		return 1.0

	prod_atoms = ATOM_PATTERN_RE.findall(partial_smiles)

	diff = abs(len(prod_atoms) - len(reac_atoms))
	return max(0.0, 1.0 - diff / len(reac_atoms))

def intermediate_smiles_reward(
	partial_smiles_list: list[str],
	reactant_smiles_list: list[str]
) -> list[float]:

	with ThreadPoolExecutor() as executor:
		args = zip(partial_smiles_list, reactant_smiles_list)
		scores = list(executor.map(lambda x: _intermediate_smiles_reward(*x), args))

	return scores

tokenizer = ChemformerTokenizerFast("bart_vocab.json")
vocab_size = tokenizer.vocab_size

collator = BARTDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

ds = USPTODataset(USPTO_CSV_FILE, tokenizer, mode="sep")
train_size = int(0.99 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size])

max_length = collator.max_length

MODEL_CONFIG = {
	"vocab_size": vocab_size,
	"d_model": 512,
	"n_layers": 6,
	"n_heads": 8,
	"d_ff": 2048,
	"dropout": 0,
	"max_seq_len": max_length,
	"aux_head": False,
}


train_dl = DataLoader(
	train_ds,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=NUM_WORKERS,
	collate_fn=collator,
)
val_dl = DataLoader(
	val_ds,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=NUM_WORKERS,
	collate_fn=collator,
)

test_dl = DataLoader(
	val_ds,
	batch_size=BATCH_SIZE,
	# batch_size=1,
	shuffle=False,
	num_workers=NUM_WORKERS,
	collate_fn=collator,
)

class AttentionPooler(nn.Module):
	def __init__(self, d_model: int):
		super().__init__()
		self.attention_projector = nn.Linear(d_model, 1)

	def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		attention_scores = self.attention_projector(hidden_states)

		if mask is not None:
			mask_transposed = mask.transpose(0, 1).unsqueeze(-1)
			attention_scores = attention_scores.masked_fill(mask_transposed, float('-inf'))

		attention_weights = F.softmax(attention_scores, dim=0)

		pooled_output = (hidden_states * attention_weights).sum(dim=0)
		return pooled_output

class Critic(nn.Module):
	def __init__(self, d_model: int, n_heads: int = 8):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads

		self.cross_attn = KVCacheMHA(
			d_model=self.d_model,
			n_heads=self.n_heads
		)

		self.pooler = AttentionPooler(self.d_model)

		self.value_head = nn.Sequential(
			nn.Linear(self.d_model, self.d_model // 2),
			nn.SiLU(),
			nn.Linear(self.d_model // 2, 1)
		)

	def forward(self, decoder_hidden_states, memory, src_mask=None):
		attn_output = self.cross_attn(
			query=decoder_hidden_states,
			key=memory,
			value=memory,
			attn_mask=src_mask,
			is_causal=False,
			kv_cache=False,
		)

		pooled_memory = self.pooler(attn_output)
		value = self.value_head(pooled_memory).squeeze(-1)

		return value
	# def forward(self, memory, src_mask=None):
	# 	pooled_memory = self.pooler(memory, src_mask)
	# 	value = self.value_head(pooled_memory).squeeze(-1)

	# 	return value



class PPOModule(pl.LightningModule):
	def __init__(
		self,
		actor: BART | Chemformer,
		critic: Critic,
		tokenizer,
		sampler_fn: Callable = beam_search_sampler,
		sampler_kwargs: Optional[Dict[str, Any]] = None,
		lr: float = 5e-4,
		ppo_epochs: int = 6,
		clip_epsilon: float = 0.2,
		ent_coef: float = 0.01,
		kl_coef: float = 0.01,
		warm_up_percent: float = 0.01,
	):
		super().__init__()
		self.save_hyperparameters(ignore=["actor", "critic"])

		self.actor = actor
		self.critic = critic
		self.tokenizer = tokenizer
		self.lr = lr

		self.total_steps = None

		self.ref_actor = copy.deepcopy(self.actor)
		for param in self.ref_actor.parameters():
			param.requires_grad = False
		self.ref_actor.eval()

		self.sampler_fn = sampler_fn
		self.sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else {}
		self.sampler_kwargs.setdefault("start_token_id", self.tokenizer.bos_token_id)
		self.sampler_kwargs.setdefault("end_token_id", self.tokenizer.eos_token_id)
		if self.sampler_fn == nucleus_sampler:
			self.sampler_kwargs.setdefault("top_p", 0.9)
		elif self.sampler_fn == beam_search_sampler:
			self.sampler_kwargs.setdefault("beam_size", 2)

		self.automatic_optimization = False

	def training_step(self, batch: dict, batch_idx: int):
		opt_actor, opt_critic = self.optimizers()

		self.actor.eval()
		self.critic.eval()

		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask")
		src_mask = src_mask.eq(0)
		tgt_tokens = batch["labels"]

		with torch.no_grad():
			memory = self.actor.encode(src_tokens, src_mask)

			self.actor.clear_cache()

			# pred_tokens, old_log_probs = self.sampler_fn(
			# 	self.actor, memory, src_mask, return_logpi=True, kv_cache=True, **self.sampler_kwargs
			# )
			# old_log_probs = old_log_probs.detach()

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
			old_log_probs = old_log_probs.detach()

			decoder_input_for_value = pred_tokens[:, :-1]
			decoder_output_for_value = self.actor.decode(
				decoder_input_for_value, memory, memory_mask=src_mask
			)

			values = self.critic(decoder_output_for_value, memory, src_mask)
			# values = self.critic(memory, src_mask)

		pred_smiles = self.tokenizer.batch_decode(pred_tokens.tolist(), skip_special_tokens=True)
		reactant_smiles = self.tokenizer.batch_decode(src_tokens.tolist(), skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)

		tanimoto = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)

		# grammar_rewards = []

		pred_tokens_cpu = pred_tokens.tolist()
		bsz, seq_len = len(pred_tokens_cpu), len(pred_tokens_cpu[0])

		all_prefixes = [
			self.tokenizer.decode(pred_tokens[b, : i + 1], skip_special_tokens=True)
			for b in range(bsz)
			for i in range(seq_len)
		]

		all_reactants = [
			reactant_smiles[b] for b in range(bsz) for _ in range(seq_len)
		]

		all_grammar_scores = intermediate_smiles_reward(all_prefixes, all_reactants)
		grammar_scores = torch.tensor(all_grammar_scores, device=self.device).view(bsz, seq_len)
		grammar = grammar_scores.mean(dim=1)

		rewards = tanimoto + grammar

		returns = rewards.detach()
		adv = (returns - values).detach()

		# Normalize advantages for training stability 
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

			kl_div = (new_log_probs - ref_log_probs).mean()
			kl_penalty = torch.abs(kl_div)

			new_values = self.critic(decoder_output.detach(), memory.detach(), src_mask)
			# new_values = self.critic(memory.detach(), src_mask)
			new_log_probs = new_log_probs.to(old_log_probs.device)

			# Policy Loss (Clipped Surrogate Objective)
			log_ratio = new_log_probs - old_log_probs
			# log_ratio_std = log_ratio.std()
			# log_ratio_mean = log_ratio.mean()

			log_ratio = torch.clamp(
				log_ratio, 
				min=-5,
				max=5,
				# min=log_ratio_mean - 3 * log_ratio_std,
				# max=log_ratio_mean + 3 * log_ratio_std
			)

			ratio = log_ratio.exp().to(adv.device)
			print(f"Ratio (actual): mean={ratio.mean():.4f}, std={ratio.std():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}")
			# ratio = (new_log_probs - old_log_probs).exp().to(adv.device)
			s1 = ratio * adv
			s2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * adv
			policy_loss = -torch.min(s1, s2).mean()

			actor_loss = policy_loss - self.hparams.ent_coef * entropy.mean() + self.hparams.kl_coef * kl_penalty

			# Value Function Loss
			value_loss = F.mse_loss(new_values, returns)

			opt_actor.zero_grad()
			opt_critic.zero_grad()

			actor_loss.backward()
			value_loss.backward()

			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

			opt_actor.step()
			opt_critic.step()

		print(f"\n--- DEBUGGING at Epoch {self.current_epoch}, Batch Index {batch_idx} ---")
		print(f"Rewards:         mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
		print(f"Values (critic): mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
		print(f"Returns:         mean={returns.mean():.4f}, std={returns.std():.4f}, min={returns.min():.4f}, max={returns.max():.4f}")
		
		adv_before_norm = (returns - values).detach()
		print(f"Adv (pre-norm):  mean={adv_before_norm.mean():.4f}, std={adv_before_norm.std():.4f}, min={adv_before_norm.min():.4f}, max={adv_before_norm.max():.4f}")
		if torch.isnan(adv_before_norm).any():
			print("!!!!!! FOUND NaN in Advantage BEFORE normalization !!!!!!")

		adv_std_val = adv_before_norm.std(unbiased=False)
		if adv_std_val < 1e-8:
			print(f"!!!!!! WARNING: Advantage STD is very low ({adv_std_val:.2e}), potential division by zero !!!!!!")

		print(f"Adv (post-norm): mean={adv.mean():.4f}, std={adv.std():.4f}, min={adv.min():.4f}, max={adv.max():.4f}")
		if torch.isnan(adv).any():
			print("!!!!!! FOUND NaN in Advantage AFTER normalization !!!!!!")

		with torch.no_grad():
			self.actor.clear_cache()
			temp_new_log_probs, temp_entropy, _ = self.actor.evaluate_actions(memory, src_mask, pred_tokens, self.tokenizer.pad_token_id)

			log_temp_ratio = temp_new_log_probs - old_log_probs
			# log_temp_ratio_std = log_temp_ratio.std()
			# log_temp_ratio_mean = log_temp_ratio.mean()

			log_temp_ratio = torch.clamp(
				log_temp_ratio, 
				min=-5,
				max=5,
				# min=log_temp_ratio_mean - 3 * log_temp_ratio_std,
				# max=log_temp_ratio_mean + 3 * log_temp_ratio_std
			)

			temp_ratio = log_temp_ratio.exp().to(adv.device)

			print(f"Ratio (sampled): mean={temp_ratio.mean():.4f}, std={temp_ratio.std():.4f}, min={temp_ratio.min():.4f}, max={temp_ratio.max():.4f}")
			if torch.isinf(temp_ratio).any():
		 	 print("!!!!!! FOUND INF in Ratio calculation !!!!!!")
		print(f"--- END DEBUGGING ---")

		self.log_dict({
			"train/policy_loss": policy_loss,
			'train/actor_loss':  actor_loss,
			"train/value_loss": value_loss,
			"train/mean_reward": rewards.mean(),
			"train/entropy": entropy.mean().item(),
			"train/kl_div": kl_div.item(),
			"train/ratio_mean": ratio.mean().item()
		}, prog_bar=True, on_step=True, on_epoch=False)

	def validation_step(self, batch: dict, batch_idx: int):
		self.actor.eval()
		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask")
		src_mask = src_mask.eq(0)
		tgt_tokens = batch["labels"]

		with torch.no_grad():
			pred_tokens = self.actor.generate(
				src_tokens,
				src_mask,
				sampler=greedy_sampler,
				kv_cache=True,
				start_token_id=self.tokenizer.bos_token_id,
				end_token_id=self.tokenizer.eos_token_id,
			)

		pred_smiles = self.tokenizer.batch_decode(pred_tokens.tolist(), skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)
		
		rewards = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device)

		batch_size = len(target_smiles)
		exact_match_count = sum(p == t for p, t in zip(pred_smiles, target_smiles))
		exact_match_rate = exact_match_count / batch_size

		self.log_dict({
			"val/mean_reward": rewards.mean(),
			"val/exact_match_rate": exact_match_rate
		}, prog_bar=True, on_epoch=True, sync_dist=True)

		return rewards.mean()

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
		
		self.log("test/beam_scores_top1", beam_scores[:, 0].mean(), prog_bar=False, sync_dist=True)

		return rewards.mean()

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

def main():
	parser = argparse.ArgumentParser(description="PPO Training and Testing")
	parser.add_argument("--action", type=str, choices=['fit', 'test', 'fit_test'], default='fit_test', help="Action to perform: 'fit', 'test', or 'fit_test'.")
	parser.add_argument("--bart_ckpt_path", type=str, default=CKPT_PATH, help="Path to BART checkpoint for actor initialization.")
	parser.add_argument("--ppo_ckpt_path", type=str, default=None, help="Path to PPO checkpoint to load for testing or resuming training.")
	args = parser.parse_args()

	untrained_bart_nn_module = BART(**MODEL_CONFIG)

	print("Loading trained LightningModule from checkpoint...")
	lightning_model = BARTModel.load_from_checkpoint(
		checkpoint_path=args.bart_ckpt_path,
		model=untrained_bart_nn_module,
		tokenizer=tokenizer
	)
	print("Model loaded successfully.")

	actor = lightning_model.model
	print(actor)

	# apply_lora_to_model(actor, rank=16, alpha=8)
	# print("Actor after applying LoRA:")
	# print(actor)

	# print("Freezing the actor's encoder...")
	# for param in actor.encoder.parameters():
	# 	param.requires_grad = False
	# print("Encoder frozen.")

	critic = Critic(actor.d_model)
	print(critic)

	logger = TensorBoardLogger('lightning_logs', name='ppo')

	ckpt_exact = ModelCheckpoint(
		monitor='val/exact_match_rate', mode='max', save_top_k=1,
		dirpath="ppo_checkpoints", filename='best-exact-{epoch:02d}-{val/exact_match_rate:.4f}'
	)
	ckpt_reward = ModelCheckpoint(
		monitor='val/mean_reward', mode='max', save_top_k=1,
		dirpath="ppo_checkpoints", filename='best-reward-{epoch:02d}-{val/mean_reward:.4f}'
	)

	trainer = pl.Trainer(
		max_epochs=NUM_EPOCHS,
		logger=logger,
		callbacks=[ckpt_exact, ckpt_reward],
		accelerator='gpu' if torch.cuda.is_available() else 'cpu',
		devices=1,
	)

	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')

	if args.action == 'test':
		if not args.ppo_ckpt_path:
			print("Error: For action 'test', --ppo_ckpt_path must be provided.")
			return

		model = PPOModule.load_from_checkpoint(
			args.ppo_ckpt_path,
			actor=actor,
			critic=critic,
			tokenizer=tokenizer
		)
		trainer.test(model, dataloaders=test_dl)
	else:  # fit or fit_test
		model = PPOModule(actor, critic, tokenizer)
		trainer.fit(model, train_dl, val_dl, ckpt_path=args.ppo_ckpt_path)
		if args.action == 'fit_test':
			trainer.test(model, dataloaders=test_dl, ckpt_path='best')

if __name__ == "__main__":
	main()
