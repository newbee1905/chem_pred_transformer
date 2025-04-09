from rdkit import Chem
import sqlite3

from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset

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
		smiles_list: list[str], ids_list: list[str],
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 256,
		noise_prob: float = 0.5, span_lambda: float = 3,
		tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id"
	):
		self.smiles_list = smiles_list
		self.ids_list = ids_list

		self.tokenizer_type = tokenizer_type
		super().__init__(tokenizer, max_length, noise_prob, span_lambda)

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
		smiles_column: str = "smiles", id_column: str = "zinc_id"
	):

		super().__init__(tokenizer, max_length, noise_prob, span_lambda)
		self.split = split

		self.conn = sqlite3.connect(sqlite_db_path)
		cursor = self.conn.cursor()
		cursor.execute(
			"SELECT rowid FROM zinc WHERE type = ? ORDER BY rowid;",
			(self.split,)
		)
		self.rowids = [r[0] for r in cursor.fetchall()]

	def __len__(self):
		return len(self.rowids)

	def __getitem__(self, idx):
		rowid = self.rowids[idx]
		cursor = self.conn.cursor()
		cursor.execute(
			"SELECT smiles FROM zinc WHERE rowid = ?;",
			(rowid,)
		)
		smi = cursor.fetchone()

		sample = self.get_smi_data(smi)
		return sample

	def __del__(self):
		try:
			self.conn.close()
		except Exception:
			pass

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
