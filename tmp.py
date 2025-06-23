import copy
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
impore re

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

from typing import Optional, Callable, Dict, Any

USPTO_CSV_FILE = "USPTO_MIT.csv"
MAX_LENGTH = 282
BATCH_SIZE = 8
NUM_WORKERS = 12
NUM_EPOCHS = 10
CKPT_PATH = "train_checkpoints_server/best-checkpoint-finetune-uspto-sep-bart_small_v8-tanimoto-v8.ckpt"

VALID_SMILES_CHARS_RE = re.compile(r'[^\w\(\)\[\]=\-\#/\+\.@%0-9]')
INVALID_BOND_SEQUENCE_RE = re.compile(r'==|##|==#|##=')
ATOM_PATTERN_RE = re.compile(r'[A-Z][a-z]?')

def intermediate_smiles_reward(
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
	"dropout": 0.1,
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

	def forward(self, decoded_hidden_states, memory, src_mask=None):
		attn_output = self.cross_attention(
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

class PPOModule(pl.LightningModule):
	def __init__(
		self,
		actor: BART | Chemformer,
		critic: Critic,
		tokenizer,
		sampler_fn: Callable = nucleus_sampler,
		sampler_kwargs: Optional[Dict[str, Any]] = None,
		lr: float = 1e-5,
		ppo_epochs: int = 2,
		clip_epsilon: float = 0.2,
		vf_coef: float = 0.5,
		ent_coef: float = 0.01,
		kl_coef: float = 0.1,
		tanimoto_scale: float = 500.0, # since max length is close to 300, tanimoto only goese from 0 to 1, increase it by 500 to have more impact compare to grammar reward
	):
		super().__init__()
		self.save_hyperparameters(ignore=["actor", "critic"])

		self.actor = actor
		self.critic = critic
		self.tokenizer = tokenizer
		self.lr = lr

		self.sampler_fn = sampler_fn
		self.sampler_kwargs = sampler_kwargs if sampler_kwargs is not None else {
			"start_token_id": self.tokenizer.bos_token_id,
			"end_token_id": self.tokenizer.eos_token_id,
		}

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

			pred_tokens, old_log_probs = self.sampler_fn(
				self.actor, memory, src_mask, return_logpi=True, kv_cache=True, **self.sampler_kwargs
			)

			decoder_input_for_value = pred_tokens[:, :-1]
			decoder_output_for_value = self.actor.decode(
				decoder_input_for_value, memory, memory_mask=src_mask
			)

			values = self.critic(decoder_output_for_value, memory, src_mask)

		pred_smiles = self.tokenizer.batch_decode(pred_tokens.tolist(), skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)

		tanimoto = compute_batch_tanimoto_rewards(pred_smiles, target_smiles, device=self.device) * self.hparams.tanimoto_scale

		grammar_rewards = []
		for b in range(pred_tokens.size(0)):
			prefixes = [
				self.tokenizer.decode(
					pred_tokens[b, : i + 1].tolist(),
					skip_special_tokens=True,
				)
				for i in range(seq.size(1))
			]
			scores = [
				intermediate_smiles_reward(pref, reactant_smiles[b])
				for pref in prefixes
			]
			grammar_rewards.append(sum(scores) / len(scores))
		grammar = torch.tensor(grammar_rewards, device=self.device)

		rewards = tanimoto + grammar

		returns = rewards.detach()
		adv = (returns - values).detach()

		# Normalize advantages for training stability 
		adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

		self.actor.train()
		self.critic.train()

		actions = pred_tokens.detach()
		for _ in range(self.hparams.ppo_epochs):
			new_log_probs, entropy, decoder_output = self.actor.evaluate_actions(
				memory, expanded_src_mask, pred_tokens, self.tokenizer.pad_token_id
			)

			new_values = self.critic(decoder_output, memory.detach(), src_mask)
			new_log_probs = new_log_probs.to(old_log_probs.device)

			# Policy Loss (Clipped Surrogate Objective)
			ratio = (new_log_probs - old_log_probs).exp().to(adv.device)
			s1 = ratio * adv
			s2 = torch.clamp(ratio, 1.0 - self.hparams.clip_epsilon, 1.0 + self.hparams.clip_epsilon) * adv
			policy_loss = -torch.min(s1, s2).mean()

			kl = (old_log_probs - new_log_probs).mean()

			actor_loss = policy_loss - self.hparams.ent_coef * entropy.mean() + self.hparams.kl_coef * kl

			# Value Function Loss
			value_loss = F.mse_loss(new_values, returns)

			opt_actor.zero_grad()
			actor_loss.backward(retain_graph=True)
			opt_actor.step()

			opt_critic.zero_grad()
			value_loss.backward()
			opt_critic.step()

		self.log_dict({
			"train/policy_loss": policy_loss,
			'train/actor_loss':  actor_loss,
			"train/value_loss": value_loss,
			"train/mean_reward": rewards.mean(),
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

		beam_size = 10  # Use a beam size that allows for top-5 and top-10 calculation
		max_length = self.actor.max_seq_len

		with torch.no_grad():
			generated_tokens, beam_scores = self.actor.generate(
				src_tokens,
				src_mask,
				sampler=beam_search_sampler,
				beam_size=beam_size,
				max_length=max_length,
				kv_cache=True,
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
		opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.hparams.lr)
		opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=self.hparams.lr)
		return opt_actor, opt_critic

class GRPOModule(pl.LightningModule):
	def __init__(
		self,
		actor: BART | Chemformer,
		tokenizer,
		sampler_fn: Callable = nucleus_sampler,
		sampler_kwargs: Optional[Dict[str, Any]] = None,
		lr: float = 1e-5,
		ppo_epochs: int = 4,
		clip_epsilon: float = 0.2,
		ent_coef: float = 0.01,
		num_generations_g: int = 4,
	):
		super().__init__()
		self.save_hyperparameters(ignore=["actor", "tokenizer"])
		
		self.num_generations_g = num_generations_g

	def training_step(self, batch: dict, batch_idx: int):
		opt_actor = self.optimizers()

		self.actor.eval()
		src_tokens = batch["input_ids"]
		src_mask = batch.get("attention_mask").eq(0)
		tgt_tokens = batch["labels"]
		
		batch_size = src_tokens.size(0)

		# Shape: [B, SeqLen] -> [B * G, SeqLen]
		expanded_src_tokens = src_tokens.repeat_interleave(self.num_generations_g, dim=0)
		expanded_src_mask = src_mask.repeat_interleave(self.num_generations_g, dim=0)
		
		with torch.no_grad():
			memory = self.actor.encode(expanded_src_tokens, expanded_src_mask)
			
			# Generate G sequences for each original input
			# pred_tokens shape: [B * G, OutSeqLen]
			# old_log_probs shape: [B * G]
			pred_tokens, old_log_probs, _ = self.sampler_fn(
				self.actor, memory, expanded_src_mask, return_logpi=True, kv_cache=True, **self.sampler_kwargs
			)

		# Comparing each of the G generations to its corresponding target
		# Shape: [B, SeqLen] -> [B * G, SeqLen]
		expanded_tgt_tokens = tgt_tokens.repeat_interleave(self.num_generations_g, dim=0)

		pred_smiles = self.tokenizer.batch_decode(pred_tokens.tolist(), skip_special_tokens=True)
		target_smiles = self.tokenizer.batch_decode(expanded_tgt_tokens.tolist(), skip_special_tokens=True)

		# [B * G]
		accepted_mask = torch.tensor(
			[p == t for p, t in zip(pred_smiles, target_smiles)],
			device=self.device
		)
		
		rejected_mask = ~accepted_mask
		old_log_probs = old_log_probs.detach()

		self.actor.train()
		for _ in range(self.hparams.ppo_epochs):
			# The evaluation is done on the full [B * G] batch
			new_log_probs, entropy, decoder_output = self.actor.evaluate_actions(
				memory, expanded_src_mask, pred_tokens, self.tokenizer.pad_token_id
			)

			accepted_log_probs = new_log_probs[accepted_mask]
			accepted_loss = -accepted_log_probs.mean() if accepted_log_probs.numel() > 0 else torch.tensor(0.0, device=self.device)

			rejected_new_log_probs = new_log_probs[rejected_mask]
			rejected_old_log_probs = old_log_probs[rejected_mask]
			if rejected_new_log_probs.numel() > 0:
				ratio = (rejected_new_log_probs - rejected_old_log_probs).exp()
				rejected_loss = F.relu(ratio - (1.0 + self.hparams.clip_epsilon)).mean()
			else:
				rejected_loss = torch.tensor(0.0, device=self.device)
				
			entropy_bonus = self.hparams.ent_coef * entropy.mean()
			total_loss = accepted_loss + rejected_loss - entropy_bonus

			opt_actor.zero_grad()
			self.manual_backward(total_loss)
			opt_actor.step()

		# Logging the accepted ratio over the B*G samples
		self.log_dict({
			"train/accepted_ratio": accepted_mask.float().mean(),
		}, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

untrained_bart_nn_module = BART(**MODEL_CONFIG)

print("Loading trained LightningModule from checkpoint...")
lightning_model = BARTModel.load_from_checkpoint(
	checkpoint_path=CKPT_PATH,
	model=untrained_bart_nn_module,
	tokenizer=tokenizer
)
print("Model loaded successfully.")

actor = lightning_model.model
print(actor)


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
	limit_train_batches=0.01,
	limit_val_batches=0.01,
)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

model = PPOModule(actor, critic, tokenizer)
trainer.fit(model, train_dl, val_dl)
trainer.test(model, dataloaders=test_dl)
