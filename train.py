import os
import joblib
from joblib import parallel_config
import pickle
from io import BytesIO
from tqdm import tqdm
import pandas as pd

from trainer.bart import BARTModel
# from trainer.vae import BARTVAEModel
from models.bart import BART
from models.chemformer import Chemformer
from models.utils import DyT
from dataset.chembl import ChemBL35Dataset, ChemBL35FilteredDataset
from dataset.uspto import USPTO50KDataset
from dataset.zinc import ZincDataset, load_smiles_by_set
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer

from utils import set_seed

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

if __name__ == "__main__":
	import torch.multiprocessing as mp
	# mp.set_sharing_strategy('file_system')

	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')

	set_seed(24)

	smiles_file = "chembl_35.smi"
	smiles_joblib_file = "filtered_chembl_smiles.joblib"
	zinc_folder = "data/zinc"
	tokenizer_dir = "trained_tokenizer"
	uspto_csv = "USPTO_FULL.csv"

	uspto_df = pd.read_csv(uspto_csv)

	tokenizer = SMILESTokenizer.from_pretrained(tokenizer_dir)
	if tokenizer.mask_token is None:
		tokenizer.add_special_tokens({"mask_token": "<mask>"})

	vocab_size = tokenizer.vocab_size

	# tokenizer = ChemformerTokenizer(filename="bart_vocab.json")
	# vocab_size = len(tokenizer)

	# ds = ChemBL35FilteredDataset(smiles_joblib_file, tokenizer, max_length=256, noise_prob=0.2, span_lambda=2, tokenizer_type="hf")
	ds = USPTO50KDataset(uspto_df["reactions"], tokenizer, max_length=256, tokenizer_type="hf")
	train_size = int(0.9 * len(ds))
	val_size = len(ds) - train_size
	train_ds, val_ds = random_split(ds, [train_size, val_size])

	# data_splits = load_smiles_by_set(zinc_folder)
	#
	# print(len(data_splits["train"]["smiles"]))
	# print(len(data_splits["val"]["smiles"]))
	#
	# train_ds = ZincDataset(data_splits["train"]["smiles"], data_splits["train"]["ids"], tokenizer)
	# val_ds = ZincDataset(data_splits["val"]["smiles"], data_splits["val"]["ids"], tokenizer)
	# test_ds = ZincDataset(data_splits["test"]["smiles"], data_splits["test"]["ids"], tokenizer)
	#
	# train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=10)
	# val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=10)


	for idx in range(len(val_ds)):
		if idx > 15:
			break
		sample = val_ds[idx]
		# Decode input and label tokens for inspection:
		input_decoded = tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)
		label_decoded = tokenizer.decode(sample["labels"].tolist(), skip_special_tokens=False)

		print(f"Example {idx+1}:")
		print("Input IDs:", sample["input_ids"])
		print("Noise Mask:", sample.get("noise_mask", "Not Provided"))
		print("Decoded Input:", input_decoded)
		print("Labels:", sample["labels"])
		print("Decoded Labels:", label_decoded)
		print("-" * 50)
	
	# train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
	# val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
	train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
	val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4)

	logger = TensorBoardLogger("lightning_logs", name="pretrain_random_smiles_zinc")
	csv_logger = CSVLogger("logs", name="pretrain_random_smiles_zinc")

	d_model = 768
	model = BART(
		vocab_size=vocab_size,
		norm_layer=nn.RMSNorm,
		d_model=d_model,
		n_heads=12,
		n_layers=6,
		d_ff=d_model * 4,
		activation="swiglu",
	)
	# model.load_state_dict(torch.load("_pretrained_model.pt", weights_only=True))
	# print(model)
	module = BARTModel(model, tokenizer)

	early_stop_callback = EarlyStopping(
		monitor="val_loss",
		patience=5,
		verbose=True,
		min_delta=0.001,
		mode="min"
	)

	checkpoint_callback = ModelCheckpoint(
		monitor="val_loss",
		dirpath="train_checkpoints",
		filename="best-checkpoint",
		save_top_k=2,
		mode="min"
	)

	timer = Timer(duration="00:20:00:00")

	trainer = pl.Trainer(
		max_steps=1000000,
		# max_epochs=1000,	
		# val_check_interval=500,
		callbacks=[early_stop_callback, checkpoint_callback, timer],
		logger=[csv_logger, logger],
		precision="16-mixed",
		gradient_clip_val=1.0,
	)

	# trainer.fit(module, train_dl, val_dl, ckpt_path="train_checkpoints/best-checkpoint-v1.ckpt")
	# trainer.fit(module, dm)
	trainer.fit(module, train_dl, val_dl)
	torch.save(module.model.state_dict(), "fast_best_ckpt_bart.pth")
