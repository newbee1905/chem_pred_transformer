import pandas as pd
from rdkit import Chem

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast

import logging
from typing import List

class USPTODataset(Dataset):
	def __init__(self, uspto_csv_file: str, tokenizer, max_length: int = 256, tokenizer_type: str = "hf"):
		uspto_df = pd.read_csv(uspto_csv_file)

		self.reactions = uspto_df["reactions"]
		self.tokenizer = tokenizer
		self.max_length = max_length

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

	def __len__(self):
		return len(self.reactions)

	def encode_and_pad(self, smiles: str) -> dict:
		"""
		Tokenises and pads a SMILES string for Chemformer tokenisers.
		"""

		token_ids = self.tokenizer.encode(smiles)[0]
		token_ids = token_ids[:self.max_length]
		attn_mask = torch.ones_like(token_ids, dtype=torch.long)

		current_len = token_ids.size(0)

		if current_len < self.max_length:
			pad_token_id = self.tokenizer.vocabulary[self.tokenizer.special_tokens["pad"]]

			pad_length = self.max_length - current_len

			pad_tensor = torch.full((pad_length,), pad_token_id)
			mask_pad = torch.zeros(pad_length, dtype=torch.long)

			token_ids = torch.cat([token_ids, pad_tensor])
			attn_mask = torch.cat([attn_mask, mask_pad])

		return {
			"input_ids": token_ids,
			"attention_mask": attn_mask,
		}

	def validate_mol(self, org_smi):
		mol = Chem.MolFromSmiles(org_smi)
		if mol is None:
			smi = org_smi
		else:
			smi = Chem.MolToSmiles(mol, canonical=True)

		return smi

	def get_smi_data(self, smi):
		if self.tokenizer.bos_token is not None:
			if not smi.startswith(self.tokenizer.bos_token):
				smi = self.tokenizer.bos_token + smi
		if self.tokenizer.eos_token is not None:
			if not smi.endswith(self.tokenizer.eos_token):
				smi = smi + self.tokenizer.eos_token
		
		if getattr(self, "tokenizer_type", "hf") == "hf":
			enc_inp = self.tokenizer(
				smi,
				truncation=True,
				max_length=self.max_length,
				padding="max_length",
				return_tensors="pt"
			)

			inp_ids = enc_inp["input_ids"].squeeze(0)
			attn_mask = enc_inp["attention_mask"].squeeze(0)
		elif self.tokenizer_type == "chemformer":
			enc_inp = self.encode_and_pad(inp_smi)

			inp_ids = enc_inp["input_ids"]
			attn_mask = enc_inp["attention_mask"]
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

		return {
			"input_ids": inp_ids,
			"attention_mask": attn_mask,
		}

	def __getitem__(self, idx):
		reaction = self.reactions[idx]
		parts = reaction.split(">")
		if len(parts) == 3:
			reactants_raw = parts[0].strip()
			products_raw	= parts[2].strip()
		else:
			reactants_raw = reaction.strip()
			products_raw	= reaction.strip()

		reactants_raw = self.validate_mol(reactants_raw)
		products_raw = self.validate_mol(products_raw)

		inp = self.get_smi_data(reactants_raw)
		label = self.get_smi_data(products_raw)

		return {
			"input_ids": inp["input_ids"].squeeze(0),
			"attention_mask": inp["attention_mask"].squeeze(0),
			"labels": label["input_ids"].squeeze(0),
		}

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
