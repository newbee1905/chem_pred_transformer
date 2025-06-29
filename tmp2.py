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
BATCH_SIZE = 16
NUM_WORKERS = 10
NUM_EPOCHS = 10
CKPT_PATH = "train_checkpoints/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"

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

actor.eval()

val_iterator = iter(val_dl)
batch = next(val_iterator)

device = "cuda"

actor = actor.to(device)
actor.freqs_cis = actor.freqs_cis.to(device)
src_tokens = batch['input_ids'][0:1].to(device)
src_mask = batch['attention_mask'][0:1].eq(0).to(device)

with torch.no_grad():
	memory = actor.encode(src_tokens, src_mask)
	actor.clear_cache()
	
	# Use kv_cache=True, as you do in the trainer
	pred_tokens, log_prob_from_sampler = nucleus_sampler(
		actor, memory.detach(), src_mask, return_logpi=True, 
	)
	actor.clear_cache()

with torch.no_grad():
	memory = actor.encode(src_tokens, src_mask) 
	
	log_prob_from_evaluate, _, _ = actor.evaluate_actions(
		memory.detach(), src_mask, pred_tokens, pad_token_id=tokenizer.pad_token_id # or whatever your pad id is
	)

print(f"Log-Prob from Sampler:         {log_prob_from_sampler.item():.6f}")
print(f"Log-Prob from evaluate_actions:  {log_prob_from_evaluate.item():.6f}")

# The difference should be extremely small (due to floating point error)
assert torch.allclose(log_prob_from_sampler, log_prob_from_evaluate), "Log probabilities are not consistent!"

print("\nSUCCESS: Log probability calculations are consistent.")
