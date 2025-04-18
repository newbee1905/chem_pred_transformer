import os
import glob
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

import lightning.pytorch as pl

from typing import List, Dict, Tuple

from dataset.base import BARTDataCollator

class PretrainBARTDataCollator(BARTDataCollator):
	def __init__(self, tokenizer, max_length: int = 64, noise_prob: float = 0.5, span_lambda: int = 3):
		super().__init__(tokenizer, max_length)
	
		self.noise_prob = noise_prob
		self.span_lambda = span_lambda

	def _span_mask_tokens(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		masked_ids = token_ids.clone()
		noise_mask = torch.zeros_like(token_ids, dtype=torch.bool)

		valid_mask = token_ids != self.pad_token_id
	
		# Exclude BOS/EOS if they exist and should not be masked
		if self.tokenizer.bos_token_id is not None:
			 valid_mask[0] = False
		if self.tokenizer.eos_token_id is not None:
			non_pad_indices = (token_ids != self.pad_token_id).nonzero(as_tuple=True)[0]
			if len(non_pad_indices) > 0:
				last_non_pad_idx = non_pad_indices[-1]

				if token_ids[last_non_pad_idx] == self.tokenizer.eos_token_id:
					valid_mask[last_non_pad_idx] = False

		valid_positions = valid_mask.nonzero(as_tuple=True)[0]

		if len(valid_positions) == 0:
			return masked_ids, noise_mask

		num_valid = len(valid_positions)
		num_to_mask = max(1, int(num_valid * self.noise_prob)) # Ensure at least one token is masked if possible
		mask_start_indices = torch.randperm(num_valid)[:num_to_mask]
		mask_positions = valid_positions[mask_start_indices]

		if len(mask_positions) == 0:
			return masked_ids, noise_mask

		span_lengths = torch.poisson(
			torch.full(
				(len(mask_positions),), self.span_lambda,
				dtype=torch.float, device=token_ids.device
			)
		).long() + 1

		already_masked = torch.zeros_like(token_ids, dtype=torch.bool)
		masked_count = 0
		for i, position in enumerate(mask_positions):
			if already_masked[position]:
				continue

			span_length = span_lengths[i].item()
			end_position = min(position + span_length, len(token_ids))

			mask_indices = []
			for j in range(position, end_position):
				if valid_mask[j] and not already_masked[j]:
					mask_indices.append(j)
				else:
					break

			if mask_indices:
				 mask_indices_tensor = torch.tensor(mask_indices, device=token_ids.device, dtype=torch.long)
				 masked_ids[mask_indices_tensor] = self.mask_token_id
				 noise_mask[mask_indices_tensor] = True
				 already_masked[mask_indices_tensor] = True
				 masked_count += len(mask_indices)

		if masked_count == 0 and len(valid_positions) > 0:
			random_idx = valid_positions[torch.randint(0, len(valid_positions), (1,)).item()]
			if not already_masked[random_idx]: # Check just in case
				 masked_ids[random_idx] = self.mask_token_id
				 noise_mask[random_idx] = True

		return masked_ids, noise_mask

	def __call__(self, batch: List[Tuple[str, str]]) -> dict[str, torch.Tensor]:
		enc = super().__call__(batch)

		noisy_input_ids_list = []
		noise_mask_list = []
		for i in range(enc["input_ids"].size(0)):
			noisy_ids, n_mask = self._span_mask_tokens(enc["input_ids"][i])
			noisy_input_ids_list.append(noisy_ids)
			noise_mask_list.append(n_mask)

		noisy_input_ids_batch = torch.stack(noisy_input_ids_list, dim=0)
		noise_mask_batch = torch.stack(noise_mask_list, dim=0)

		return {
			"input_ids": noisy_input_ids_batch,
			"attention_mask": enc["attention_mask"],
			"noise_mask": noise_mask_batch, 
			"labels": enc["labels"],
			"labels_attention_mask": enc["labels_attention_mask"],
		}

class PretrainBARTDataset(Dataset):
	"""
	Base class for BART-style pretraining datasets.
	"""
	
	def __init__(self, tokenizer, n_merge: int = 0):
		self.tokenizer = tokenizer
		self.n_merge = n_merge

		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.bos_token = tokenizer.bos_token
		self.eos_token = tokenizer.eos_token

	def augment_smi(self, org_smi: str) -> Tuple[str, str]:
		mol = Chem.MolFromSmiles(org_smi)
		if mol is None:
			inp_smi = org_smi
			label_smi = org_smi
		else:
			try:
				 inp_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
			except Exception:
				 inp_smi = Chem.MolToSmiles(mol, canonical=True)

			label_smi = Chem.MolToSmiles(mol, canonical=True)

		if self.bos_token:
			if not inp_smi.startswith(self.bos_token):
				inp_smi = self.bos_token + inp_smi
			if not label_smi.startswith(self.bos_token):
				label_smi = self.bos_token + label_smi
		if self.eos_token:
			if not inp_smi.endswith(self.eos_token):
				inp_smi = inp_smi + self.eos_token
			if not label_smi.endswith(self.eos_token):
				label_smi = label_smi + self.eos_token

		return inp_smi, label_smi

	def __getitem__(self, idx):
		if self.n_merge < 1:
			org_smi = self.smiles_list[idx]
		else:
			idx = idx * self.n_merge
			org_smi = ".".join(self.smiles_list[idx:idx + self.n_merge])

		inp_smi, label_smi = self.augment_smi(smi)
		
		return inp_smi, label_smi

	def __len__(self):
		if self.n_merge < 1:
			return len(self.smiles_list)
		return len(self.smiles_list) // self.n_merge
