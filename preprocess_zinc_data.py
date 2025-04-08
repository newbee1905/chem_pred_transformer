import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os import path
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
mp.set_start_method("fork")
from dataset.zinc import load_smiles_by_set
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tokenisers.neocart import SMILESTokenizer

CHUNK_SIZE = 9999997
tokenizer_dir = "trained_tokenizer"
tokenizer = SMILESTokenizer.from_pretrained(tokenizer_dir)
if tokenizer.mask_token is None:
	tokenizer.add_special_tokens({"mask_token": "<mask>"})

vocab_size = tokenizer.vocab_size
zinc_folder = "data/zinc"
data_splits = load_smiles_by_set(zinc_folder)
num_workers = 10

tokenizer_mp = None
def init_pool(_tokenizer):
	global tokenizer_mp
	tokenizer_mp = _tokenizer
	
def token_length(smi):
	global tokenizer_mp
	token_ids = tokenizer_mp.encode(smi, add_special_tokens=True)
	return len(token_ids)

def process_chunk(smiles_chunk, worker_pool, chunk_idx, set_type):
	"""Process a chunk of SMILES and update histogram data"""

	lengths = list(
		tqdm(
			worker_pool.imap(token_length, smiles_chunk),
			total=len(smiles_chunk),
			desc=f"Computing chunk {chunk_idx} for {set_type}"
		)
	)
	
	max_length = max(lengths) if lengths else 0
	
	hist_data = {}
	for length in lengths:
		hist_data[length] = hist_data.get(length, 0) + 1
	
	chunk_file = f"hist_data_{set_type}_chunk_{chunk_idx}.pkl"
	with open(chunk_file, "wb") as f:
		pickle.dump({"max_length": max_length, "hist_data": hist_data}, f)
	
	return max_length

for set_type in ["train", "test", "val"]:
	smiles_list = data_splits[set_type]["smiles"]
	print(f"Processing {set_type} split with {len(smiles_list)} molecules.")
	
	global_max_length = 0
	chunk_files = []
	
	with Pool(processes=num_workers, initializer=init_pool, initargs=(tokenizer,)) as pool:
		for i in range(0, len(smiles_list), CHUNK_SIZE):
			chunk = smiles_list[i:i+CHUNK_SIZE]
			chunk_idx = i // CHUNK_SIZE
			
			chunk_max_length = process_chunk(chunk, pool, chunk_idx, set_type)
			global_max_length = max(global_max_length, chunk_max_length)
			
			chunk_file = f"hist_data_{set_type}_chunk_{chunk_idx}.pkl"
			chunk_files.append(chunk_file)
	
	print(f"Maximum token length in {set_type} split: {global_max_length}")
	
	combined_hist_data = {}
	for chunk_file in chunk_files:
		with open(chunk_file, "rb") as f:
			chunk_data = pickle.load(f)
			for length, count in chunk_data["hist_data"].items():
				combined_hist_data[length] = combined_hist_data.get(length, 0) + count
	
	plt.figure(figsize=(10, 6))
	
	lengths = sorted(combined_hist_data.keys())
	counts = [combined_hist_data[length] for length in lengths]
	
	plt.bar(lengths, counts, width=1, edgecolor="black")
	plt.title(f"Token Length Distribution for {set_type} Split")
	plt.xlabel("Token Length")
	plt.ylabel("Frequency")
	plt.grid(axis='y', alpha=0.75)
	
	histogram_filename = f"token_length_histogram_{set_type}.png"
	plt.savefig(histogram_filename)
	plt.close()
	print(f"Saved histogram for {set_type} split to {histogram_filename}.")
	
	hist_data_filename = f"token_length_histogram_data_{set_type}.pkl"
	with open(hist_data_filename, "wb") as f:
		pickle.dump({
			"max_length": global_max_length,
			"hist_data": combined_hist_data
		}, f)
	print(f"Saved histogram data for {set_type} split to {hist_data_filename}.")
	
	for chunk_file in chunk_files:
		os.remove(chunk_file)
	print(f"Removed {len(chunk_files)} temporary chunk files.")
