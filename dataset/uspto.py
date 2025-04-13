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
from itertools import permutations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
from queue import Empty
import multiprocessing as mp

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

		reactants_raw = validate_mol(reactants_raw)
		products_raw = validate_mol(products_raw)

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

		return [
			f"{'.'.join(r_perm)}>>{'.'.join(p_perm)}"
			for r_perm in reactant_perms
			for p_perm in product_perms
		]

	except Exception as e:
		print(f"Error processing: {reaction_smiles}\n{e}")
		return []

def producer(reactions, permutation_queue, producer_done_event, pbar):
	"""Permute reactions and add them to the queue."""
	try:
		for reaction in reactions:
			permutations = permute_reaction(reaction)
			for permuted in permutations:
				permutation_queue.put(permuted)
				pbar.update(1)
	finally:
		producer_done_event.set()

def consumer(permutation_queue, output_folder, idx_counter, consumer_done_event, producer_done_event):
	"""Consume permutations from the queue and write to LMDB."""
	env = lmdb.open(
		os.path.join(output_folder, "permuted_uspto.lmdb"),
		map_size=1099511627776,  # 1TB
		max_dbs=1
	)
	
	try:
		while not (producer_done_event.is_set() and permutation_queue.empty()):
			try:
				permuted = permutation_queue.get(timeout=1)
				
				with env.begin(write=True) as txn:
					with idx_counter.get_lock():
						idx = idx_counter.value
						idx_counter.value += 1
					
					txn.put(f"{idx}".encode("ascii"), pickle.dumps(permuted))
				
				permutation_queue.task_done()
				
			except Empty:
				continue
	finally:
		consumer_done_event.set()
		env.sync()
		env.close()

def finalize_lmdb(output_folder, total_count):
	"""Add the length record to the LMDB file."""
	env = lmdb.open(
		os.path.join(output_folder, "permuted_uspto.lmdb"),
		map_size=1099511627776,  # 1TB
	)

	with env.begin(write=True) as txn:
		txn.put(b"__len__", pickle.dumps(total_count))

	env.sync()
	env.close()


def preprocess_uspto_lmdb(uspto_csv_file, output_folder: str, num_workers: int = None, max_queue_size: int = 10000):
	"""Load SMILES data from CSV files and permute them and save into lmdb."""

	if num_workers is None:
		num_workers = max(1, mp.cpu_count() - 1)

	uspto_df = pd.read_csv(uspto_csv_file)
	reactions = uspto_df["reactions"]

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	env = lmdb.open(
		os.path.join(output_folder, "permuted_uspto.lmdb"),
		map_size=1099511627776, # 1TB
	)

	permutation_queue = mp.JoinableQueue(maxsize=max_queue_size)
	producer_done_event = mp.Event()
	consumer_done_events = [mp.Event() for _ in range(num_workers)]
	idx_counter = mp.Value('i', 0)
	
	total_estimated_perms = len(reactions) * 16
	pbar = tqdm(total=total_estimated_perms, desc="Processing Permutations")

	producer_process = mp.Process(
		target=producer, 
		args=(reactions, permutation_queue, producer_done_event, pbar)
	)
	producer_process.start()

	consumer_processes = []
	for i in range(num_workers):
		process = mp.Process(
			target=consumer,
			args=(permutation_queue, output_folder, idx_counter, consumer_done_events[i], producer_done_event)
		)
		process.start()
		consumer_processes.append(process)
	
	producer_process.join()
	
	for process in consumer_processes:
		process.join()
	
	total_count = idx_counter.value
	finalize_lmdb(output_folder, total_count)
	
	print(f"Total permutations processed: {total_count}")
