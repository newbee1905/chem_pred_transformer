from rdkit import Chem

from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset
import lightning.pytorch as pl

import os
# import pandas as pd
import fireducks.pandas as pd
import numpy as np
from glob import glob
from os import path
from tqdm import tqdm
from io import BytesIO
import lmdb

import pickle

def load_smiles_by_set(folder: str, smiles_column="smiles", id_column="zinc_id", set_column="set"):
	"""Load SMILES data from CSV files and organize them into train, val, and test lists."""
	csv_files = sorted(glob(path.join(folder, "*.csv")))

	data = {
		"train": {"smiles": [], "ids": []}, 
		"val": {"smiles": [], "ids": []}, 
		"test": {"smiles": [], "ids": []},
	}

	for file in tqdm(csv_files, desc="Reading ZINC CSV files"):
		chunks = pd.read_csv(file, usecols=[smiles_column, id_column, set_column], chunksize=10000)
		for chunk in tqdm(chunks, desc=f"Processing chunks in {path.basename(file)}", leave=False):
			for set_type in ["train", "val", "test"]:
				mask = chunk[set_column] == set_type
				data[set_type]["smiles"].extend(chunk.loc[mask, smiles_column].tolist())
				data[set_type]["ids"].extend(chunk.loc[mask, id_column].tolist())

	return data

def serialize_entry(idx, zinc_id, smiles, compress=3):
	key = f"{idx:08d}".encode("ascii")
	buffer = BytesIO()
	pickle.dump(smiles, buffer)

	return key, buffer.getvalue()

def create_lmdb_for_set(set_data: dict, lmdb_path: str, n_jobs=10):
	"""
	Create an LMDB database file for a specific dataset split.
	"""

	env = lmdb.open(lmdb_path, map_size=10**12)
	with env.begin(write=True) as txn:
		for idx, (zinc_id, smiles) in tqdm(enumerate(zip(set_data["ids"], set_data["smiles"])), total=len(set_data["ids"])):
			with parallel_config(backend='threading', n_jobs=n_jobs):
				key, value = serialize_entry(idx, zinc_id, smiles)
			txn.put(key, value)
	env.close()

class ZincDataset(PretrainBARTDataset):
	"""
	Zinc dataset that uses Tokyo Cabinet for key-value storage.
	If a db_path is provided, data is fetched on-demand from the Tokyo Cabinet DB.
	"""

	def __init__(
		self, 
		tokenizer,
		max_length: int = 512,
		noise_prob: float = 0.5,
		tokenizer_type: str = "hf",
		smiles_list: list[str] = None,
		db_path: str = None,
	):

		self.tokenizer_type = tokenizer_type
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.noise_prob = noise_prob
		self.db_path = db_path

		if db_path is not None:
			self._init_lmdb()
		else:
			if smiles_list is None:
				raise ValueError("Provide smiles_list if not using LMDB.")

			self.smiles_list = smiles_list
			self._length = len(smiles_list)

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

	def _init_lmdb(self):
		self.db_env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False)
		with self.db_env.begin() as txn:
			self._length = txn.stat()['entries']

	def __getitem__(self, idx):
		if self.db_path is None:
			org_smi = self.smiles_list[idx]
		else:
			if self.db_env is None:
				self._init_lmdb()
			with self.db_env.begin() as txn:
				key = f"{idx:08d}".encode("ascii")
				value = txn.get(key)
			if value is None:
				raise IndexError("Index out of range in LMDB DB")
			org_smi = pickle.load(BytesIO(value))

		return self.get_smi_data(org_smi)

	def __len__(self):
		return self._length

	def __getstate__(self):
		state = self.__dict__.copy()
		# Do not pickle the LMDB environment
		state['db_env'] = None

		return state

	def __setstate__(self, state):
		self.__dict__.update(state)

		if self.db_path is not None and self.db_env is None:
			self._init_lmdb()

class ZincDataModule(pl.LightningDataModule):
	def __init__(
		self,
		csv_folder: str,
		lmdb_dir: str,
		tokenizer,
		batch_size: int = 64,
		num_workers: int = 4,
		max_length: int = 512,
		noise_prob: float = 0.5,
		tokenizer_type: str = "hf"
	):
		super().__init__()
		self.csv_folder = csv_folder
		self.lmdb_dir = lmdb_dir
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.max_length = max_length
		self.noise_prob = noise_prob
		self.tokenizer_type = tokenizer_type

		self.train_db_path = os.path.join(lmdb_dir, "train.lmdb")
		self.val_db_path = os.path.join(lmdb_dir, "val.lmdb")
		self.test_db_path = os.path.join(lmdb_dir, "test.lmdb")

	def prepare_data(self):
		if not (os.path.exists(self.train_db_path) and os.path.exists(self.val_db_path) and os.path.exists(self.test_db_path)):
			data_by_set = load_smiles_by_set(self.csv_folder)
			os.makedirs(self.lmdb_dir, exist_ok=True)
			create_lmdb_for_set(data_by_set["train"], self.train_db_path)
			create_lmdb_for_set(data_by_set["val"], self.val_db_path)
			create_lmdb_for_set(data_by_set["test"], self.test_db_path)

	def setup(self, stage=None):
		self.train_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.train_db_path
		)
		print(self.train_dataset)
		self.val_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.val_db_path
		)
		self.test_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.test_db_path
		)

	def train_dataloader(self):
		print(len(self.train_dataset))
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			shuffle=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers
		)
