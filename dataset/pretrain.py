import os
import glob
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class PretrainBARTDataset(Dataset):
	"""
	Base class for BART-style pretraining datasets.
	"""
	
	def __init__(self, tokenizer, max_length: int = 64, noise_prob: float = 0.5):
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.noise_prob = noise_prob

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

	def __getitem__(self, idx):
		org_smi = self.smiles_list[idx]
		mol = Chem.MolFromSmiles(org_smi)
		if mol is None:
			inp_smi = org_smi
			label_smi = org_smi
		else:
			inp_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=True)
			label_smi = Chem.MolToSmiles(mol, canonical=True)
		
		# Add bos and eos tokens if available.
		if self.tokenizer.bos_token is not None:
			if not inp_smi.startswith(self.tokenizer.bos_token):
				inp_smi = self.tokenizer.bos_token + inp_smi
			if not label_smi.startswith(self.tokenizer.bos_token):
				label_smi = self.tokenizer.bos_token + label_smi
		if self.tokenizer.eos_token is not None:
			if not inp_smi.endswith(self.tokenizer.eos_token):
				inp_smi = inp_smi + self.tokenizer.eos_token
			if not label_smi.endswith(self.tokenizer.eos_token):
				label_smi = label_smi + self.tokenizer.eos_token
		
		if getattr(self, "tokenizer_type", "hf") == "hf":
			enc_inp = self.tokenizer(
				inp_smi,
				truncation=True,
				max_length=self.max_length,
				padding="max_length",
				return_tensors="pt"
			)
			enc_label = self.tokenizer(
				label_smi,
				truncation=True,
				max_length=self.max_length,
				padding="max_length",
				return_tensors="pt"
			)

			inp_ids = enc_inp["input_ids"].squeeze(0)
			attn_mask = enc_inp["attention_mask"].squeeze(0)
			labels = enc_label["input_ids"].squeeze(0)
		elif self.tokenizer_type == "chemformer":
			enc_inp = self.encode_and_pad(inp_smi)
			enc_label = self.encode_and_pad(label_smi)

			inp_ids = enc_inp["input_ids"]
			attn_mask = enc_inp["attention_mask"]
			labels = enc_label["input_ids"]
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

		non_pad_indices = (attn_mask == 1).nonzero(as_tuple=True)[0]
		num_to_mask = max(1, int(len(non_pad_indices) * self.noise_prob))
		mask_indices = torch.randperm(len(non_pad_indices))[:num_to_mask]
		selected_mask_positions = non_pad_indices[mask_indices]

		noisy_inp_ids = inp_ids.clone()
		noisy_inp_ids[selected_mask_positions] = self.tokenizer.mask_token_id

		return {
			"input_ids": noisy_inp_ids,
			"attention_mask": attn_mask,
			"labels": labels
		}

	def __len__(self):
		return len(self.smiles_list)
