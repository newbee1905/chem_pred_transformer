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

class USPTODataset(Dataset):
	def __init__(self, uspto_csv_file: str, tokenizer: PreTrainedTokenizerFast, mode: str = "mixed"):
		uspto_df = pd.read_csv(uspto_csv_file)
		self.reactions = uspto_df["reactions"]
		self.mode = mode

		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.bos_token = tokenizer.bos_token
		self.eos_token = tokenizer.eos_token

	def __len__(self):
		return len(self.reactions)

	def __getitem__(self, idx):
		reaction = self.reactions[idx]

		parts = reaction.split(">")
		if len(parts) == 3:
			reactants_raw = parts[0].strip()
			catalyst_raw = parts[1].strip()
			products_raw	= parts[2].strip()

			if self.mode == "sep":
				if len(catalyst_raw) > 0 and catalyst_raw != "":
					reactants_raw = f"{reactants_raw}>{catalyst_raw}"
			else:
				new_reactants_raw = f"{reactants_raw}.{catalyst_raw}"
				try:
					mol = Chem.MolFromSmiles(new_reactants_raw)
					if mol is not None:
						reactants_raw = Chem.MolToSmiles(new_reactants_raw, canonical=True)
				except:
					pass
		else:
			reactants_raw = reaction.strip()
			products_raw	= reaction.strip()

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

class USPTORetrosynthesisDataset(USPTODataset):
	def __init__(self, uspto_csv_file: str, tokenizer, max_length: int = 256, tokenizer_type: str = "hf"):
		super().__init__(uspto_csv_file, tokenizer, max_length, tokenizer_type)

	def __len__(self):
		return len(self.reactions) * 2

	def __getitem__(self, idx):
		reaction = self.reactions[idx // 2]
		parts = reaction.split(">")
		if len(parts) == 3:
			reactants_raw = parts[0].strip()
			products_raw	= parts[2].strip()
		else:
			reactants_raw = reaction.strip()
			products_raw	= reaction.strip()

		inp = self.get_smi_data(reactants_raw)
		label = self.get_smi_data(products_raw)

		if i % 2 == 1:
			inp, label = label, inp

		return {
			"input_ids": inp["input_ids"].squeeze(0),
			"attention_mask": inp["attention_mask"].squeeze(0),
			"labels": label["input_ids"].squeeze(0),
		}

class USPTOChemformerDataset(USPTODataset):
	def __init__(self, uspto_df, tokenizer, max_length: int = 256, tokenizer_type: str = "hf"):
		self.uspto_df = uspto_df.reset_index()
		self.tokenizer = tokenizer
		self.max_length = max_length

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

	def __len__(self):
		return len(self.uspto_df)

	def __getitem__(self, idx):
		reactants_mol, products_mol = self.uspto_df.loc[idx, ["reactants_mol", "products_mol"]]

		reactants_smi = Chem.MolToSmiles(reactants_mol, canonical=True)
		products_smi = Chem.MolToSmiles(products_mol, canonical=True)

		inp = self.get_smi_data(reactants_smi)
		label = self.get_smi_data(products_smi)

		return {
			"input_ids": inp["input_ids"].squeeze(0),
			"attention_mask": inp["attention_mask"].squeeze(0),
			"labels": label["input_ids"].squeeze(0),
		}

def permute_reaction(reaction_smiles):
	try:
		parts = reaction_smiles.strip().split(">")
		if len(parts) != 3:
			return []

		reactants, _, products = parts
		reactant_mols = reactants.split(".")
		product_mols = products.split(".")

		reactant_perms = list(permutations(reactant_mols))
		product_perms = list(permutations(product_mols))

		print(reactant_perms, product_perms)

		return [
			f"{'.'.join(r_perm)}>>{'.'.join(p_perm)}"
			for r_perm in reactant_perms
			for p_perm in product_perms
		]

	except Exception as e:
		print(f"Error processing: {reaction_smiles}\n{e}")
		return []

def preprocess_uspto_lmdb(uspto_csv_file, output_folder: str):
	"""Load SMILES data from CSV files and permute them and save into lmdb."""

	uspto_df = pd.read_csv(uspto_csv_file)
	reactions = uspto_df["reactions"]

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	with Pool(processes=cpu_count() - 1) as pool:
		permuted_reactions = list(tqdm(pool.imap(permute_reaction, reactions), total=len(reactions)))

	print(len(permuted_reactions))
	write_lmdb(permuted_reactions, f"{output_folder}/permuted_uspto.lmdb")
