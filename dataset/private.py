import os
import pandas as pd
from rdkit import Chem

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast

import logging
from typing import List

import lmdb
from utils import validate_mol, write_lmdb
from multiprocessing import Pool, cpu_count
from itertools import permutations
from tqdm import tqdm

class PrivateDataset(Dataset):
	def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast, rotate: bool = False):
		df = pd.read_csv(path)
		self.reactants_catalyst = df["Input"]
		self.products = df["Output"]
		self.rotate = rotate 

		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.bos_token = tokenizer.bos_token
		self.eos_token = tokenizer.eos_token

	def __len__(self):
		return len(self.products)

	def __getitem__(self, idx):
		reactants_raw = self.reactants_catalyst[idx]
		products_raw = self.products[idx]

		reactant, catalyst = reactants_raw.split(">")

		# reactant_mol = Chem.MolFromSmiles(reactant)
		# catalyst_mol = Chem.MolFromSmiles(catalyst)
		# product_mol = Chem.MolFromSmiles(products_raw)

		# reactant = Chem.MolToSmiles(reactant_mol, canonical=True)
		# catalyst = Chem.MolToSmiles(catalyst_mol, canonical=True)
		# products_raw = Chem.MolToSmiles(product_mol, canonical=True)

		if self.rotate:
			reactants_raw = f"{catalyst}>{reactant}"
		else:
			reactants_raw = f"{reactant}>{catalyst}"

		if self.bos_token:
			if not reactants_raw.startswith(self.bos_token):
				reactants_raw = self.bos_token + reactants_raw
			if not products_raw.startswith(self.bos_token):
				products_raw = self.bos_token + products_raw
		if self.eos_token:
			if not reactants_raw.endswith(self.eos_token):
				reactants_raw = reactants_raw + self.eos_token
			if not products_raw.endswith(self.eos_token):
				products_raw = products_raw + self.eos_token

		return reactants_raw, products_raw
