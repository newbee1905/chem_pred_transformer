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
		folder: str,
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer, max_length: int = 512,
		noise_prob: float = 0.5, tokenizer_type: str = "hf",
		smiles_column: str = "smiles", id_column: str = "zinc_id"
	):
		csv_files = sorted(glob(path.join(folder, "*.csv")))

		smiles_chunks = []
		ids_chunks = []

		for file in tqdm(csv_files, desc="Reading ZINC csv file"):
			chunks = pd.read_csv(file, usecols=[smiles_column, id_column], chunksize=10000)
			for chunk in tqdm(chunks, desc=f"Processing chunks in {path.basename(file)}", leave=False):
				smiles_chunks.append(chunk[smiles_column].values)
				ids_chunks.append(chunk[id_column].values)

		self.smiles_list = np.concatenate(smiles_chunks)
		self.ids_list = np.concatenate(ids_chunks)

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
