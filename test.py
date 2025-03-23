from trainer.bart import BARTModel
# from trainer.vae import BARTVAEModel
from models.bart import BART
from models.utils import DyT
from dataset.chembl import ChemBL35Dataset, ChemformerChemBL35Dataset
from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
	import torch.multiprocessing as mp
	# mp.set_sharing_strategy('file_system')

	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')

	smiles_file = "chembl_35.smi"
	# tokenizer_dir = "trained_tokenizer"
	#
	# tokenizer = SMILESTokenizer.from_pretrained(tokenizer_dir)
	# if tokenizer.mask_token is None:
	# 	tokenizer.add_special_tokens({"mask_token": "<mask>"})
	tokeniser = ChemformerTokenizer(filename="bart_vocab.json")

	ds = ChemformerChemBL35Dataset(smiles_file, tokeniser, max_length=256, noise_prob=0.5)
	train_size = int(0.9 * len(ds))
	val_size = len(ds) - train_size
	train_ds, val_ds = random_split(ds, [train_size, val_size])

	train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=10)
	val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=10)

	logger = TensorBoardLogger("lightning_logs", name="pretrain_random_smiles_chembl35")

	model = BART(
		vocab_size=len(tokeniser),
		norm_layer=nn.LayerNorm,
		d_model=512,
		n_heads=8,
		n_layers=6,
		d_ff=2048,
		activation="gelu",
	)
	model.load_state_dict(torch.load("chemformer_small.pt", weights_only=True))
	module = BARTModel(model, tokeniser)

	early_stop_callback = EarlyStopping(
		monitor="val_loss",
		patience=10,
		verbose=True,
		mode="min"
	)

	checkpoint_callback = ModelCheckpoint(
		monitor="val_loss",
		dirpath="train_checkpoints",
		filename="best-checkpoint",
		save_top_k=1,
		mode="min"
	)

	trainer = pl.Trainer(
		max_epochs=1,	
		callbacks=[early_stop_callback, checkpoint_callback],
		logger=logger,
		limit_test_batches=0.01,
	)

	# trainer.fit(module, train_dl, val_dl, ckpt_path="train_checkpoints/best-checkpoint-v1.ckpt")
	# trainer.fit(module, train_dl, val_dl)
	# torch.save(module.model.state_dict(), "chemformer_small.pth")

	trainer.test(module, val_dl)
