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
from joblib import parallel_config
import pickle
import lmdb
import bz2

from utils import chunks

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

def serialize_entry(idx, zinc_id, smiles):
	key = f"{idx:08d}".encode("ascii")

	return key, pickle.dumps(smiles)

def create_lmdb_for_set(set_data: dict, lmdb_path: str, n_jobs=10):
	"""
	Create an LMDB database file for a specific dataset split.
	"""

	env = lmdb.open(lmdb_path, map_size=10**12)

	with env.begin(write=True) as txn:
		for idx, (zinc_id, smiles) in tqdm(
			enumerate(zip(set_data["ids"], set_data["smiles"])),
			total=len(set_data["ids"]),
		):
			with parallel_config(backend='threading', n_jobs=n_jobs):
				key, value = serialize_entry(idx, zinc_id, smiles)

			txn.put(key, value)

	env.close()

def create_lmdb_chunks_for_set(set_data: dict, lmdb_path: str, max_rows: int = 10000000, n_jobs: int = 10):
	"""
	Create an LMDB database file for a specific dataset split.
	"""

	for chunk_idx, set_chunk in tqdm(
		enumerate(chunks(zip(set_data["ids"], set_data["smiles"]), max_rows)),
		desc="Processing chunks",
		total=(len(set_data["ids"]) // max_rows),
	):
		chunk_lmdb_path = f"{lmdb_path}_chunk_{chunk_idx}"
		env = lmdb.open(chunk_lmdb_path, map_size=10**11)

		with env.begin(write=True) as txn:
			for idx, (zinc_id, smiles) in tqdm(enumerate(set_chunk), total=len(set_chunk)):
				with parallel_config(backend='threading', n_jobs=n_jobs):
					key, value = serialize_entry(max_rows * chunk_idx + idx, zinc_id, smiles)

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
		chunked: bool = False,
		max_rows_per_chunk: int = 10000000,
	):
		self.tokenizer_type = tokenizer_type
		self.tokenizer = tokenizer

		self.max_length = max_length
		self.noise_prob = noise_prob

		self.db_path = db_path
		self.chunked = True
		self.max_rows_per_chunk = max_rows_per_chunk

		if db_path is not None:
			if self.chunked:
				self._init_lmdb_chunked()
			else:
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

	def _init_lmdb_chunked(self):
		self.db_envs = []

		chunk_files = sorted(glob(f"{self.db_path}_chunk_*"))
		self._length = 0

		for chunk_file in chunk_files:
			env = lmdb.open(chunk_file, readonly=True, lock=False, readahead=False)
			with env.begin() as txn:
				n_entries = txn.stat()['entries']

			self.db_envs.append(env)
			self._length += n_entries

	def __getitem__(self, idx):
		if self.db_path is None:
			org_smi = self.smiles_list[idx]
		else:
			if self.chunked:
				chunk_idx = idx // self.max_rows_per_chunk
				env = self.db_envs[chunk_idx]
				with env.begin() as txn:
					key = f"{idx:08d}".encode("ascii")
					value = txn.get(key)

				if value is None:
					raise IndexError("Index out of range in chunked LMDB")

				org_smi = pickle.load(BytesIO(value))
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
		# Do not pickle LMDB environments.
		if self.chunked:
			state['db_envs'] = None
		else:
			state['db_env'] = None
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		if self.db_path is not None:
			if self.chunked and (not hasattr(self, 'db_envs') or self.db_envs is None):
				self._init_lmdb_chunked()
			elif not self.chunked and (not hasattr(self, 'db_env') or self.db_env is None):
				self._init_lmdb()

class ZincDataModule(pl.LightningDataModule):
	def __init__(
		self,
		csv_folder: str,
		lmdb_dir: str,
		tokenizer,
		batch_size: int = 64,
		n_workers: int = 4,
		max_length: int = 512,
		noise_prob: float = 0.5,
		tokenizer_type: str = "hf",
		train_chunked: bool = False,
		val_chunked: bool = False,
		test_chunked: bool = False,
		max_rows_per_chunk: int = 10000000,
	):
		super().__init__()
		self.csv_folder = csv_folder
		self.lmdb_dir = lmdb_dir

		self.tokenizer = tokenizer
		self.max_length = max_length
		self.noise_prob = noise_prob
		self.tokenizer_type = tokenizer_type

		self.batch_size = batch_size
		self.n_workers = n_workers

		self.train_chunked = train_chunked
		self.val_chunked = val_chunked
		self.test_chunked = test_chunked
		self.max_rows_per_chunk = max_rows_per_chunk

		self.train_db_path = os.path.join(lmdb_dir, "train.lmdb")
		self.val_db_path = os.path.join(lmdb_dir, "val.lmdb")
		self.test_db_path = os.path.join(lmdb_dir, "test.lmdb")

	def _check_split_exists(self, db_path: str, chunked: bool):
		if chunked:
			chunk_files = glob(f"{db_path}_chunk_*")
			return len(chunk_files) > 0
		return os.path.exists(db_path)

	def prepare_data(self):
		data_by_set = None
		os.makedirs(self.lmdb_dir, exist_ok=True)

		if not self._check_split_exists(self.train_db_path, self.train_chunked):
			if data_by_set is None:
				data_by_set = load_smiles_by_set(self.csv_folder)
			if self.train_chunked:
				create_lmdb_chunks_for_set(data_by_set["train"], self.train_db_path, max_rows=self.max_rows_per_chunk)
			else:
				create_lmdb_for_set(data_by_set["train"], self.train_db_path)

		if not self._check_split_exists(self.val_db_path, self.val_chunked):
			if data_by_set is None:
				data_by_set = load_smiles_by_set(self.csv_folder)
			if self.val_chunked:
				create_lmdb_chunks_for_set(data_by_set["val"], self.val_db_path, max_rows=self.max_rows_per_chunk)
			else:
				create_lmdb_for_set(data_by_set["val"], self.val_db_path)

		if not self._check_split_exists(self.test_db_path, self.test_chunked):
			if data_by_set is None:
				data_by_set = load_smiles_by_set(self.csv_folder)
			if self.test_chunked:
				create_lmdb_chunks_for_set(data_by_set["test"], self.test_db_path, max_rows=self.max_rows_per_chunk)
			else:
				create_lmdb_for_set(data_by_set["test"], self.test_db_path)

	def setup(self, stage=None):
		self.train_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.train_db_path,
			chunked=self.train_chunked,
			max_rows_per_chunk=self.max_rows_per_chunk,
		)
		self.val_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.val_db_path,
			chunked=self.val_chunked,
			max_rows_per_chunk=self.max_rows_per_chunk,
		)
		self.test_dataset = ZincDataset(
			tokenizer=self.tokenizer,
			max_length=self.max_length,
			noise_prob=self.noise_prob,
			tokenizer_type=self.tokenizer_type,
			db_path=self.test_db_path,
			chunked=self.test_chunked,
			max_rows_per_chunk=self.max_rows_per_chunk,
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			num_workers=self.n_workers,
			shuffle=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			num_workers=self.n_workers
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			num_workers=self.n_workers
		)
