import argparse

from tokenisers.neochem import ChemformerTokenizerFast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset.zinc import load_smiles_by_set
from dataset.uspto import USPTODataset
from dataset.private import PrivateDataset
from dataset.base import BARTDataCollator

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
from models.chemformer import Chemformer

from models.lora import apply_lora_to_model
from models.rl import Critic
from trainer.ppo import PPOModule

USPTO_CSV_FILE = "USPTO_MIT.csv"
PRIVATE_DATA_PATH = "data/private"
PRIVATE_DATA_ROTATE = False
# MAX_LENGTH = 282
MAX_LENGTH = 256
BATCH_SIZE = 64
NUM_WORKERS = 16
NUM_EPOCHS = 10
CKPT_PATH = "train_checkpoints_server/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
# CKPT_PATH = "train_checkpoints/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"

tokenizer = ChemformerTokenizerFast("bart_vocab.json")
vocab_size = tokenizer.vocab_size

collator = BARTDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

# ds = USPTODataset(USPTO_CSV_FILE, tokenizer, mode="sep")
# train_size = int(0.99 * len(ds))
# val_size = len(ds) - train_size
# train_ds, val_ds = random_split(ds, [train_size, val_size])
# test_ds = val_ds

train_ds = PrivateDataset(f"{PRIVATE_DATA_PATH}/train.csv", tokenizer, PRIVATE_DATA_ROTATE)
val_ds = PrivateDataset(f"{PRIVATE_DATA_PATH}/val.csv", tokenizer, PRIVATE_DATA_ROTATE)
test_ds = PrivateDataset(f"{PRIVATE_DATA_PATH}/test.csv", tokenizer, PRIVATE_DATA_ROTATE)

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
	test_ds,
	batch_size=BATCH_SIZE,
	# batch_size=1,
	shuffle=False,
	num_workers=NUM_WORKERS,
	collate_fn=collator,
)

def save_ppo_model_to_pth(ckpt_path, output_path, model_config, tokenizer):
	"""
	Loads a PPOModule from a .ckpt file and saves only the actor's weights
	to a .pth file.
	"""
	print(f"Loading PPOModule from checkpoint: {ckpt_path}")

	actor_scaffold = BART(**model_config)
	critic_scaffold = Critic(actor_scaffold.d_model)

	ppo_model = PPOModule.load_from_checkpoint(
		checkpoint_path=ckpt_path,
		actor=actor_scaffold,
		critic=critic_scaffold,
		tokenizer=tokenizer,
		strict=True
	)
	print("Model loaded successfully from checkpoint.")

	actor_state_dict = ppo_model.actor.state_dict()

	torch.save(actor_state_dict, output_path)
	print(f"Actor weights saved to: {output_path}")

def main():
	parser = argparse.ArgumentParser(description="PPO Training and Testing")
	parser.add_argument("--action", type=str, choices=['fit', 'test', 'fit_test', 'save_pth'], default='fit_test', help="Action to perform: 'fit', 'test', or 'fit_test'.")
	parser.add_argument("--bart_ckpt_path", type=str, default=CKPT_PATH, help="Path to BART checkpoint for actor initialization.")
	parser.add_argument("--ppo_ckpt_path", type=str, default=None, help="Path to PPO checkpoint to load for testing or resuming training.")
	parser.add_argument("--output_pth_path", type=str, default="ppo_model_weights.pth", help="Path to save the final .pth weights file.")
	parser.add_argument("--is_per_step", type=bool, default=False, help="Switch to use GAE.")
	args = parser.parse_args()

	if args.action == 'save_pth':
		if not args.ppo_ckpt_path:
			print("Error: For action 'save_pth', --ppo_ckpt_path must be provided.")
			return

		save_ppo_model_to_pth(args.ppo_ckpt_path, args.output_pth_path, MODEL_CONFIG, tokenizer)
		return

	untrained_bart_nn_module = BART(**MODEL_CONFIG)

	if args.ppo_ckpt_path:
		print(f"A PPO checkpoint is provided. Actor architecture will be initialized for checkpoint loading: {args.ppo_ckpt_path}")
		actor = BART(**MODEL_CONFIG)
	else:
		if not args.bart_ckpt_path:
			raise ValueError("To start a new PPO training run, --bart_ckpt_path must be provided.")

		print(f"Initializing new PPO run from BART checkpoint: {args.bart_ckpt_path}")
		untrained_bart_nn_module = BART(**MODEL_CONFIG)
		lightning_model = BARTModel.load_from_checkpoint(
			checkpoint_path=args.bart_ckpt_path,
			model=untrained_bart_nn_module,
			tokenizer=tokenizer
		)
		actor = lightning_model.model
		print("Actor initialized successfully from BART checkpoint.")
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
	ckpt_reward = ModelCheckpoint(
		monitor='v_tanimoto', mode='max', save_top_k=1,
		dirpath="ppo_checkpoints", filename='best-reward-{epoch:02d}-{v_tanimoto:.4f}'
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
		if args.ppo_ckpt_path:
			print(f"Loading model weights from PPO checkpoint for a new training run: {args.ppo_ckpt_path}")
			model = PPOModule.load_from_checkpoint(
				checkpoint_path=args.ppo_ckpt_path,
				actor=actor,
				critic=critic,
				tokenizer=tokenizer
			)
		else:
			print("Creating a new PPOModule for training from scratch (using BART weights).")
			model = PPOModule(actor, critic, tokenizer, is_per_step=args.is_per_step)

		trainer.fit(model, train_dl, val_dl, ckpt_path=args.ppo_ckpt_path)
		if args.action == 'fit_test':
			trainer.test(model, dataloaders=test_dl, ckpt_path='best')

if __name__ == "__main__":
	main()
