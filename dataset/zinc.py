from rdkit import Chem

from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset

import pandas as pd
# import fireducks.pandas as pd
import numpy as np
from glob import glob
from os import path
from tqdm import tqdm

class ZincDataset(PretrainBARTDataset):
	"""Zinc dataset that reads SMILES strings and molecule IDs from CSV files in a folder."""
	
	def __init__(
		self,
		smiles_list: list[str], ids_list: list[str],
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 512,
		noise_prob: float = 0.5, tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id"
	):
		self.smiles_list = smiles_list
		self.ids_list = ids_list

		self.tokenizer_type = tokenizer_type
		super().__init__(tokenizer, max_length, noise_prob)

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

	def __getitem__(self, idx):
		sample = super().__getitem__(idx)
		sample["id"] = self.ids_list[idx]

		return sample

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
