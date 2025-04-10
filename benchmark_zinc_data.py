from rdkit import Chem
import sqlite3
import lmdb
import pickle

import time

from tokenisers.neocart import SMILESTokenizer
from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset
from dataset.zinc import ZincDataset, ZincLazyDataset, ZincLMDBDataset, ZincNMAPDataset, load_smiles_by_set, preprocess_zinc_data_splits_nmap

import os
import pandas as pd
# import fireducks.pandas as pd
import numpy as np
from glob import glob
from os import path
from tqdm import tqdm

import sys

def estimate_memory(smiles_list):
	total = sum(sys.getsizeof(s) for s in smiles_list)
	print(f"Total memory usage (approx): {total / (1024 ** 2):.2f} MB")

zinc_folder = "data/zinc"
lmdb_zinc_folder = "data/zinc/preprocessed"

tokenizer = SMILESTokenizer.from_pretrained("trained_tokenizer")

if tokenizer.mask_token is None:
	tokenizer.add_special_tokens({"mask_token": "<mask>"})

vocab_size = tokenizer.vocab_size

data_splits = load_smiles_by_set(zinc_folder)

# estimate_memory(data_splits["train"]["smiles"])

# preprocess_zinc_data_splits_nmap(data_splits, lmdb_zinc_folder)

val_ds = ZincDataset(data_splits["val"]["smiles"], max_length=36, noise_prob=0.2, span_lambda=2, tokenizer=tokenizer, tokenizer_type="hf")
val_sqlite_ds = ZincLazyDataset(f"{zinc_folder}/zinc_processed.db", split="val", max_length=36, noise_prob=0.2, span_lambda=2, tokenizer=tokenizer, tokenizer_type="hf")
val_lmdb_ds = ZincLMDBDataset(f"{lmdb_zinc_folder}/val.lmdb", max_length=36, noise_prob=0.2, span_lambda=2, tokenizer=tokenizer, tokenizer_type="hf")
val_nmap_ds = ZincNMAPDataset(f"{lmdb_zinc_folder}/val.nmap", max_length=36, noise_prob=0.2, span_lambda=2, tokenizer=tokenizer, tokenizer_type="hf")

val_dl = DataLoader(
	val_ds,
	batch_size=2,
	shuffle=False,
	num_workers=2,
)

val_sqlite_dl = DataLoader(
	val_sqlite_ds,
	batch_size=2,
	shuffle=False,
	num_workers=2,
)

val_lmdb_dl = DataLoader(
	val_lmdb_ds,
	batch_size=2,
	shuffle=False,
	num_workers=2,
)

val_nmap_dl = DataLoader(
	val_nmap_ds,
	batch_size=2,
	shuffle=False,
	num_workers=2,
)

def benchmark_data_loader(data_loader, name, n_iter=3):
  total_time = 0
  for iteration in range(n_iter):
    start = time.time()
    # Iterate over the entire DataLoader.
    for _ in data_loader:
      pass
    elapsed = time.time() - start
    print(f"{name} iteration {iteration+1}: {elapsed:.3f} seconds")
    total_time += elapsed
  avg_time = total_time / n_iter
  print(f"{name} average: {avg_time:.3f} seconds")
  return avg_time

benchmark_data_loader(val_dl, "Normal")
benchmark_data_loader(val_sqlite_dl, "SQLite")
benchmark_data_loader(val_lmdb_dl, "LMDB")
benchmark_data_loader(val_nmap_dl, "Numpy Memmap")
