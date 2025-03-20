from rdkit import Chem

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast

import logging
from typing import List

class USPTO50KTorchDataset(Dataset):
	def __init__(self, reactions: List[str], spe_tokenizer: PreTrainedTokenizerFast):
		# Expect reaction SMILES in the X field, formatted as "reactants>reagents>products".
		self.reactions = reactions
		self.spe_tokenizer = spe_tokenizer

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

		molecule = reactants_raw if idx % 2 == 0 else products_raw
		enc = self.spe_tokenizer(molecule)

		return {
			"input_ids": enc["input_ids"].squeeze(0),
			"attention_mask": enc["attention_mask"].squeeze(0),
			"labels": enc["input_ids"].squeeze(0),
		}
