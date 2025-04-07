import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tokenisers.neocart import SMILESTokenizer
from tokenisers.chemformer import ChemformerTokenizer

from torch.utils.data import DataLoader, random_split
from dataset.chembl import ChemBL35Dataset, ChemBL35FilteredDataset
from dataset.uspto import USPTODataset, USPTORetrosynthesisDataset
from dataset.zinc import ZincDataset, load_smiles_by_set

import importlib
import pickle
import lightning.pytorch as pl
from utils import set_seed, filter_none_kwargs

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
		tokenizer = ChemformerTokenizer(filename=cfg.tokenizer.path)
		vocab_size = len(tokenizer)
	else:
		raise ValueError(f"Tokenizer {cfg.tokenizer.type} is not supported")

	if cfg.dataset.get("split_mode") == "predefined":
		del cfg.dataset.split_mode

		if cfg.dataset.type == "zinc":
			data_splits = load_smiles_by_set(cfg.dataset.path)

			del cfg.dataset.type
			del cfg.dataset.path

			train_ds = instantiate(cfg.dataset, data_splits["train"]["smiles"], data_splits["train"]["ids"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, data_splits["val"]["smiles"], data_splits["val"]["ids"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, data_splits["test"]["smiles"], data_splits["test"]["ids"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
		else:
			with open(cfg.dataset.path, "rb") as f:
				data_splits = pickle.load(f)

			del cfg.dataset.type
			del cfg.dataset.path

			train_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "train"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			val_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "valid"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			# test_ds = instantiate(cfg.dataset, data_splits[data_splits["set"] == "test"], tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
			test_ds = instantiate(cfg.dataset, data_splits, tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)

		max_length = train_ds.max_length

		test_dl = DataLoader(
			test_ds,
			batch_size=1,
			shuffle=False,
			num_workers=cfg.num_workers
		)
	else:
		del cfg.dataset.type
		ds = instantiate(cfg.dataset, tokenizer=tokenizer, tokenizer_type=cfg.tokenizer.type)
		train_size = int(cfg.train_split * len(ds))
		val_size = len(ds) - train_size
		train_ds, val_ds = random_split(ds, [train_size, val_size])

		max_length = ds.max_length

		test_dl = DataLoader(
			val_ds,
			batch_size=1,
			shuffle=False,
			num_workers=cfg.num_workers
		)

	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers
	)
	val_dl = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers
	)

	model = instantiate(cfg.model, vocab_size=vocab_size)
	if "pretrained_state_dict" in cfg:
		model.load_state_dict(torch.load(cfg.pretrained_state_dict))
	print(model)
	module = instantiate(cfg.module, model, tokenizer, max_length=max_length, mode=cfg.task_type)
	print(module)

	callbacks = []
	if "callbacks" in cfg:
		for cb_cfg in cfg.callbacks:
			callbacks.append(instantiate(cb_cfg))

	loggers = []
	if "loggers" in cfg:
		for logger_cfg in cfg.loggers:
			loggers.append(instantiate(logger_cfg))
	
	trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=loggers)

	trainer_kwargs = filter_none_kwargs(ckpt_path=cfg.get("ckpt_path"))

	if cfg.task == "test":
		trainer.test(module, test_dl, **trainer_kwargs)
	else:
		trainer.fit(module, train_dl, val_dl, **trainer_kwargs)

if __name__ == "__main__":
	from rdkit import RDLogger
	RDLogger.DisableLog('rdApp.*')

	OmegaConf.register_new_resolver("py", resolve_py_path)

	my_app()
