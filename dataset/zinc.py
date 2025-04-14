from rdkit import Chem
import sqlite3
import lmdb
import pickle

from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset

from utils import write_memmap, read_memmap, write_lmdb

import os
import pandas as pd
# import fireducks.pandas as pd
import numpy as np
from glob import glob
from os import path
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import multiprocessing as mp

class ZincDataset(PretrainBARTDataset):
	"""Zinc dataset that reads SMILES strings and molecule IDs from CSV files in a folder."""
	
	def __init__(
		self,
		smiles_list: list[str],
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 256,
		noise_prob: float = 0.5, span_lambda: float = 3,
		tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id",
		**kwargs,
	):
		self.smiles_list = smiles_list

		self.tokenizer_type = tokenizer_type
		super().__init__(tokenizer, max_length, noise_prob, span_lambda, **kwargs)

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

class ZincLazyDataset(PretrainBARTDataset):
	"""
	Lazy-loading ZINC dataset backed by SQLite for a single split.
	"""
	def __init__(
		self,
		sqlite_db_path: str,
		split: str,
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 256,
		noise_prob: float = 0.5, span_lambda: float = 3,
		tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id",
		**kwargs,
	):
		self.tokenizer_type = tokenizer_type
		super().__init__(tokenizer, max_length, noise_prob, span_lambda, **kwargs)

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

		self.split = split

		self.conn = sqlite3.connect(sqlite_db_path)

		self.conn.execute("PRAGMA journal_mode = WAL")
		self.conn.execute("PRAGMA synchronous = NORMAL")
		self.conn.execute("PRAGMA cache_size = 100000")
		self.conn.execute("PRAGMA temp_store = MEMORY")
		self.conn.execute("PRAGMA mmap_size = 4000000000") 

		cursor = self.conn.cursor()
		cursor.execute(
			"SELECT rowid FROM zinc WHERE type = ? ORDER BY rowid;",
			(self.split,)
		)
		self.rowids = [r[0] for r in cursor.fetchall()]

		self.cursor = self.conn.cursor()

	def __len__(self):
		return len(self.rowids)

	def __getitem__(self, idx):
		rowid = self.rowids[idx]
		self.cursor.execute(
			"SELECT smiles FROM zinc WHERE rowid = ?;",
			(rowid,)
		)
		smi = self.cursor.fetchone()[0]

		sample = self.get_smi_data(smi)
		return sample

	def __del__(self):
		try:
			self.conn.close()
		except Exception:
			pass

class ZincLMDBDataset(PretrainBARTDataset):
	"""
	Lazy-loading ZINC dataset backed by LMDB.
	"""
	def __init__(
		self,
		lmdb_path: str,
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 256,
		noise_prob: float = 0.5, span_lambda: float = 3,
		tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id",
		**kwargs,
	):
		self.tokenizer_type = tokenizer_type
		super().__init__(tokenizer, max_length, noise_prob, span_lambda, **kwargs)
		self.lmdb_path = lmdb_path
		self.env = None

		with lmdb.open(
			self.lmdb_path,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False
		) as env:
			with env.begin() as txn:
				self.length = pickle.loads(txn.get(b"__len__"))

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

	def __len__(self):
		if self.n_merge < 1:
			return self.length
		return self.length // self.n_merge

	def __getitem__(self, idx):
		# Lazy initialisation: If self.env is not defined in this worker, open it.
		# each worker creates is own env instead of the env being forked around
		if self.env is None:
			self.env = lmdb.open(
				self.lmdb_path,
				readonly=True,
				lock=False,
				readahead=False,
				meminit=False
			)

		with self.env.begin(write=False) as txn:
			if self.n_merge < 1:
				key = f"{idx}".encode("ascii")
				smi = pickle.loads(txn.get(key))
			else:
				idx = idx * self.n_merge
				ids = list(range(idx, idx + self.n_merge))
				keys = [f"{i}".encode("ascii") for i in ids]
				smis = [pickle.loads(txn.get(key)) for key in keys]
				smi = ".".join(smis)

		return self.get_smi_data(smi)

class ZincNMAPDataset(PretrainBARTDataset):
	"""
	Lazy-loading ZINC dataset backed by Numpy NMAP.
	"""
	def __init__(
		self,
		nmap_path: str,
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 256,
		noise_prob: float = 0.5, span_lambda: float = 3,
		tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id"
	):
		super().__init__(tokenizer, max_length, noise_prob, span_lambda)
		self.data = read_memmap(nmap_path) 

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		smi = self.data[idx].decode('utf-8')
		return self.get_smi_data(smi)

def process_file(file_path, smiles_column="smiles", id_column="zinc_id", set_column="set"):
	"""
	Process a single CSV file and extract SMILES strings, IDs, and set split.
	"""
	local_data = {"smiles": [], "ids": [], "set": []}
	chunks = pd.read_csv(
		file_path,
		usecols=[smiles_column, id_column, set_column],
		chunksize=100000,
	)

	for chunk in tqdm(chunks, desc=f"Processing chunks in {path.basename(file_path)}", leave=False):
		for set_type in ["train", "val", "test"]:
			mask = chunk[set_column] == set_type
			local_data["smiles"].extend(chunk.loc[mask, smiles_column].tolist())
			local_data["ids"].extend(chunk.loc[mask, id_column].tolist())
			local_data["set"].extend([set_type] * mask.sum())

	return local_data

def load_smiles_by_set(folder: str, smiles_column="smiles", id_column="zinc_id", set_column="set"):
	"""Load SMILES data from CSV files and organize them into train, val, and test lists."""
	csv_files = sorted(glob(path.join(folder, "*.csv")))

	data = {
		"train": {"smiles": [], "ids": []}, 
		"val": {"smiles": [], "ids": []}, 
		"test": {"smiles": [], "ids": []},
	}

	for file in tqdm(csv_files, desc="Reading ZINC CSV files"):
		chunks = pd.read_csv(file, usecols=[smiles_column, id_column, set_column], chunksize=100000)
		for chunk in tqdm(chunks, desc=f"Processing chunks in {path.basename(file)}", leave=False):
			for set_type in ["train", "val", "test"]:
				mask = chunk[set_column] == set_type
				data[set_type]["smiles"].extend(chunk.loc[mask, smiles_column].tolist())
				data[set_type]["ids"].extend(chunk.loc[mask, id_column].tolist())

	return data

def preprocess_zinc_data_splits(data_split, output_folder: str, chunk_size: int = 9999997):
	"""Load SMILES data from CSV files and organize them into train, val, and test lists."""

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	train_smiles = data_split["train"]["smiles"]
	train_ids = data_split["train"]["ids"]
	num_train = len(train_smiles)
	print(f"Found {num_train} training records. Saving in chunks of {chunk_size} records each.")

	for i in range(0, num_train, chunk_size):
		chunk_smiles = train_smiles[i : i + chunk_size]
		chunk_ids = train_ids[i : i + chunk_size]
		df_train = pd.DataFrame({"smiles": chunk_smiles, "ids": chunk_ids})
		chunk_index = i // chunk_size
		out_file = path.join(output_folder, f"train_chunk_{chunk_index}.csv")
		df_train.to_csv(out_file, index=False)

	val_smiles = data_split["val"]["smiles"]
	val_ids = data_split["val"]["ids"]
	test_smiles = data_split["test"]["smiles"]
	test_ids = data_split["test"]["ids"]

	df_val = pd.DataFrame({"smiles": val_smiles, "id": val_ids})
	out_val = path.join(output_folder, "val.csv")
	df_val.to_csv(out_val, index=False)

	df_test = pd.DataFrame({"smiles": test_smiles, "id": test_ids})
	out_test = path.join(output_folder, "test.csv")
	df_test.to_csv(out_test, index=False)

def preprocess_zinc_sqlite(input_folder):
	"""
	Preprocess ZINC CSV files using multiprocessing and save all data into one SQLite table.
	"""

	csv_files = sorted(glob(path.join(input_folder, "*.csv")))

	with Pool(processes=(cpu_count() - 1)) as pool:
		tasks = [(file_path) for file_path in csv_files]
		results = pool.map(process_file, tasks)

	db_path = path.join(input_folder, "zinc_processed.db")
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	cursor.execute("""
		CREATE TABLE IF NOT EXISTS zinc (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			smiles TEXT,
			zinc_id TEXT,
			type TEXT	
		);
	""")
	cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON zinc(type);")

	for data in results:
		records = list(zip(data["smiles"], data["ids"], data["set"]))
		print(f"Inserting {len(records)} records into the database.")
		cursor.executemany("INSERT INTO zinc (smiles, zinc_id, type) VALUES (?, ?, ?);", records)

	conn.commit()
	conn.close()
	print("Preprocessing complete. SQLite DB saved at:", db_path)

def preprocess_zinc_data_splits_lmdb(data_split, output_folder: str):
	"""Load SMILES data from CSV files and organize them into train, val, and test lmdb."""

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	train_smiles = data_split["train"]["smiles"]
	val_smiles = data_split["val"]["smiles"]
	test_smiles = data_split["test"]["smiles"]

	write_lmdb(train_smiles, f"{output_folder}/train.lmdb")
	write_lmdb(val_smiles, f"{output_folder}/val.lmdb")
	write_lmdb(test_smiles, f"{output_folder}/test.lmdb")

def preprocess_zinc_data_splits_nmap(data_split, output_folder: str):
	"""Load SMILES data from CSV files and organize them into train, val, and test memory map."""

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	train_smiles = data_split["train"]["smiles"]
	val_smiles = data_split["val"]["smiles"]
	test_smiles = data_split["test"]["smiles"]

	write_memmap(train_smiles, f"{output_folder}/train.nmap")
	write_memmap(val_smiles, f"{output_folder}/val.nmap")
	write_memmap(test_smiles, f"{output_folder}/test.nmap")
