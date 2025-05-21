import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer
from tokenisers.neochem import ChemformerTokenizerFast

import torch
from torch.utils.data import DataLoader, random_split
from dataset.zinc import load_smiles_by_set

import importlib
import pickle
import lightning.pytorch as pl
from utils import set_seed, filter_none_kwargs

import torch._dynamo
from torch._dynamo import disable

from lightning.pytorch.callbacks import ModelCheckpoint

torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.cache_size_limit = 256
pl.LightningModule.log = disable(pl.LightningModule.log)

def resolve_py_path(path: str):
	module_name, class_name = path.rsplit(".", 1)
	module = importlib.import_module(module_name)
	return getattr(module, class_name)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))

	set_seed(cfg.seed)

	if cfg.tokenizer.type == "hf":
		tokenizer = SMILESTokenizer.from_pretrained(cfg.tokenizer.path)

		if tokenizer.mask_token is None:
			tokenizer.add_special_tokens({"mask_token": "<mask>"})

		vocab_size = tokenizer.vocab_size
	elif cfg.tokenizer.type == "chemformer":
		tokenizer = ChemformerTokenizerFast(cfg.tokenizer.path)
		# tokenizer = ChemformerTokenizer(filename=cfg.tokenizer.path)
		vocab_size = tokenizer.vocab_size
	else:
		raise ValueError(f"Tokenizer {cfg.tokenizer.type} is not supported")

	if cfg.dataset.get("split_mode") == "predefined":
		del cfg.dataset.split_mode

		if cfg.dataset.type == "zinc":
			data_splits = load_smiles_by_set(cfg.dataset.path)

			del cfg.dataset.type
			del cfg.dataset.path

			train_ds = instantiate(cfg.dataset, data_splits["train"]["smiles"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, data_splits["val"]["smiles"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, data_splits["test"]["smiles"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
		elif cfg.dataset.type == "zinc_sqlite":
			del cfg.dataset.type

			train_ds = instantiate(cfg.dataset, split="train", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, split="val", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, split="test", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
		elif cfg.dataset.type == "zinc_lmdb":
			del cfg.dataset.type
			lmdb_folder = cfg.dataset.lmdb_folder
			del cfg.dataset.lmdb_folder

			collator = instantiate(cfg.dataset.collator, tokenizer=tokenizer)
			del cfg.dataset.collator

			train_ds = instantiate(cfg.dataset, f"{lmdb_folder}/train.lmdb", tokenizer=tokenizer)
			val_ds = instantiate(cfg.dataset, f"{lmdb_folder}/val.lmdb", tokenizer=tokenizer)
			test_ds = instantiate(cfg.dataset, f"{lmdb_folder}/test.lmdb", tokenizer=tokenizer)
		elif cfg.dataset.type == "zinc_nmap":
			del cfg.dataset.type
			nmap_folder = cfg.dataset.nmap_folder
			del cfg.dataset.nmap_folder

			train_ds = instantiate(cfg.dataset, f"{nmap_folder}/train.nmap", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, f"{nmap_folder}/val.nmap", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, f"{nmap_folder}/test.nmap", tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
		else:
			with open(cfg.dataset.path, "rb") as f:
				data_splits = pickle.load(f)

			del cfg.dataset.type
			del cfg.dataset.path

			train_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "train"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "valid"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			# test_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "test"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, data_splits, tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)

		max_length = collator.max_length

		test_dl = DataLoader(
			test_ds,
			batch_size=cfg.batch_size,
			shuffle=False,
			num_workers=cfg.num_workers,
			collate_fn=collator,
		)
	else:
		del cfg.dataset.type

		collator = instantiate(cfg.dataset.collator, tokenizer=tokenizer)
		del cfg.dataset.collator

		ds = instantiate(cfg.dataset, tokenizer=tokenizer)
		train_size = int(cfg.train_split * len(ds))
		val_size = len(ds) - train_size
		train_ds, val_ds = random_split(ds, [train_size, val_size])

		max_length = collator.max_length

		test_dl = DataLoader(
			val_ds,
			batch_size=cfg.batch_size,
			# batch_size=1,
			shuffle=False,
			num_workers=cfg.num_workers,
			collate_fn=collator,
		)

	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		collate_fn=collator,
	)
	val_dl = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		collate_fn=collator,
	)

	model = instantiate(cfg.model, vocab_size=vocab_size, max_seq_len=max_length, max_batch_size=cfg.batch_size)
	if "pretrained_state_dict" in cfg:
		model.load_state_dict(torch.load(cfg.pretrained_state_dict), strict=False)
	print(model)
	module_kwargs = filter_none_kwargs(
		kv_cache=cfg.get("kv_cache"),
	)
	module = instantiate(cfg.module, model, tokenizer, mode=cfg.task_type, **module_kwargs)

	callbacks = []
	if "callbacks" in cfg:
		for cb_cfg in cfg.callbacks:
			callbacks.append(instantiate(cb_cfg))

	loggers = []
	if "loggers" in cfg:
		for logger_cfg in cfg.loggers:
			loggers.append(instantiate(logger_cfg))

	trainer_kwargs = filter_none_kwargs(
		devices=cfg.get("devices"),
		num_nodes=cfg.get("num_nodes"),
		num_sanity_val_steps=cfg.get("num_sanity_val_steps"),
	)

	trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=loggers, **trainer_kwargs)

	trainer_kwargs = filter_none_kwargs(
		ckpt_path=cfg.get("ckpt_path"),
		sampler=cfg.get("sampler"),
	)

	if cfg.task == "test":
		trainer.test(module, test_dl, **trainer_kwargs)
	else:
		trainer.fit(module, train_dl, val_dl, **trainer_kwargs)
		if "pth_path" in cfg:
			ckpt_cb = next(
				cb for cb in trainer.callbacks
				if isinstance(cb, ModelCheckpoint)
			)
			best_ckpt = ckpt_cb.best_model_path
			sd = torch.load(best_ckpt, map_location="cpu")["state_dict"]
			torch.save(sd, cfg.pth_path)
			print(f"Best model weights saved to {cfg.pth_path} (from {best})")

if __name__ == "__main__":
	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')

	OmegaConf.register_new_resolver("py", resolve_py_path)

	my_app()
