from trainer.bart import BARTModel
# from trainer.vae import BARTVAEModel
from models.bart import BART
from models.chemformer import Chemformer
from models.utils import DyT
from dataset.chembl import ChemBL35Dataset
from dataset.zinc import ZincDataset 
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
	zinc_folder = "data/zinc"
	tokenizer_dir = "trained_tokenizer"

	# tokeniser = SMILESTokenizer.from_pretrained(tokenizer_dir)
	# if tokeniser.mask_token is None:
	# 	tokeniser.add_special_tokens({"mask_token": "<mask>"})
	tokenizer = ChemformerTokenizer(filename="bart_vocab.json")

	ds = ChemBL35Dataset(smiles_file, tokenizer, max_length=512, noise_prob=0.5, span_lambda=1, tokenizer_type="chemformer")
	# ds = ChemBL35Dataset(smiles_file, tokeniser, max_length=512, noise_prob=0.5, tokenizer_type="hf")
	train_size = int(0.9 * len(ds))
	val_size = len(ds) - train_size
	train_ds, val_ds = random_split(ds, [train_size, val_size])

	train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=10)
	val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=10)

	logger = TensorBoardLogger("lightning_logs", name="pretrain_random_smiles_chembl35")

	# model = Chemformer(
	# 	vocab_size=ds.vocab_size,
	# 	norm_layer=nn.LayerNorm,
	# 	d_model=512,
	# 	n_heads=8,
	# 	n_layers=6,
	# 	d_ff=2048,
	# 	max_seq_len=512,
	# 	activation="gelu",
	# )
	# model.load_state_dict(torch.load("chemformer_small_2.pth", weights_only=True))
	model = BART(
		vocab_size=ds.vocab_size,
		norm_layer=nn.RMSNorm,
		d_model=512,
		n_heads=8,
		n_layers=6,
		d_ff=2048,
		max_seq_len=512,
		activation="swiglu",
	)
	module = BARTModel(model, tokenizer)

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
		limit_test_batches=0.0001,
	)

	# trainer.fit(module, train_dl, val_dl, ckpt_path="train_checkpoints/best-checkpoint-v1.ckpt")
	# trainer.fit(module, train_dl, val_dl)
	# torch.save(module.model.state_dict(), "chemformer_small.pth")

	# trainer.test(module, val_dl)
	trainer.test(module, val_dl, ckpt_path="train_checkpoints/best-checkpoint-v4.ckpt")
