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
	
	def __init__(self, tokenizer, max_length: int = 64, noise_prob: float = 0.5, span_lambda: float = 3.0, n_merge: int = 0):
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.noise_prob = noise_prob
		self.span_lambda = span_lambda
		self.n_merge = n_merge

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

	def span_mask_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
		"""
		Applies span masking to the token_ids.
		Iterates over the sequence and, with probability noise_prob, masks a contiguous span.
		"""

		masked_ids = token_ids.clone()
		noise_mask = torch.zeros_like(token_ids, dtype=torch.bool)
		pad_id = self.tokenizer.pad_token_id
		mask_id = self.tokenizer.mask_token_id

		length = masked_ids.size(0)
		i = 0
		masked = False

		while i < length:
			if token_ids[i] == pad_id:
				break

			if torch.rand(1).item() < self.noise_prob:
				span_length = max(
					1,
					int(torch.poisson(
						torch.tensor(self.span_lambda, dtype=torch.float, device=token_ids.device)
					).item())
				)

				# Move end pointer forward, but stop if we hit a pad token
				end = i
				count = 0
				while end < length and count < span_length and token_ids[end] != pad_id:
					end += 1
					count += 1
				
				if end > i:
					masked_ids[i:end] = mask_id
					noise_mask[i:end] = True
					masked = True

				i = end
			else:
				i += 1

		if not masked:
			valid_indices = (token_ids != pad_id).nonzero(as_tuple=False).view(-1)

			if valid_indices.numel() > 0:
				rand_idx = valid_indices[torch.randint(0, valid_indices.numel(), (1,)).item()]
				masked_ids[rand_idx] = mask_id
				noise_mask[rand_idx] = True 

		return masked_ids, noise_mask

	def get_smi_data(self, org_smi):
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

		noisy_inp_ids, noise_mask = self.span_mask_tokens(inp_ids)

		return {
			"input_ids": noisy_inp_ids,
			"attention_mask": attn_mask,
			"noise_mask": noise_mask,
			"labels": labels
		}

	def __getitem__(self, idx):
		if self.n_merge < 1:
			org_smi = self.smiles_list[idx]
		else:
			idx = idx * 4
			org_smi = ".".join(self.smiles_list[idx:idx+4])
		return self.get_smi_data(org_smi)

	def __len__(self):
		if self.n_merge < 1:
			return len(self.smiles_list)
		return len(self.smiles_list) // self.n_merge
