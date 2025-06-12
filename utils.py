import itertools
import os
import random
import numpy as np
import torch
import lmdb
import pickle

from rdkit import Chem

def validate_mol(org_smi):
	mol = Chem.MolFromSmiles(org_smi)
	if mol is None:
		smi = org_smi
	else:
		smi = Chem.MolToSmiles(mol, canonical=True)

	return smi

def chunks(iterable, chunk_size):
	it = iter(iterable)
	while True:
		chunk = list(itertools.islice(it, chunk_size))
		if not chunk:
			break
		yield chunk

def set_seed(seed: int = 24) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)

def filter_none_kwargs(**kwargs):
	return {k: v for k, v in kwargs.items() if v is not None}

def is_valid_smiles(smi):
	try:
		from rdkit import Chem
		return Chem.MolFromSmiles(smi) is not None
	except ImportError:
		return True

def write_memmap(arr, output_file, max_length=100):
	n = len(arr)

	memmap_array = np.memmap(output_file, dtype=f'S{max_length}', mode='w+', shape=(n,))

	for i, s in enumerate(arr):
		encoded = s.encode('utf-8')[:max_length]
		memmap_array[i] = encoded

	memmap_array.flush()
	return memmap_array

def read_memmap(output_file, max_length=100):
	return np.memmap(output_file, dtype=f'S{max_length}', mode='r')

def write_lmdb(smiles_list, lmdb_path):
	env = lmdb.open(lmdb_path, map_size=1099511627776) # 1TB max size
	with env.begin(write=True) as txn:
		for idx, smi in enumerate(smiles_list):
			txn.put(f"{idx}".encode("ascii"), pickle.dumps(smi))
		txn.put(b"__len__", pickle.dumps(len(smiles_list)))
	env.sync()
	env.close()

def _expand_for_k(
	memory: torch.Tensor,
	src_mask: torch.Tensor,
	k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Return *views* (no copy) that repeat each batch row `k` times."""

	# (seq, bsz, 1, H) -> (seq, bsz, k, d_model) -> (seq, bsz * k, d_model)
	mem_k = (
		memory
			.unsqueeze(2)
			.expand(-1, -1, k, -1)
			.reshape(memory.size(0), -1, memory.size(-1))
	)

	# (bsz, 1, seq) -> (bsz, k, seq) -> (bsz * k, seq)
	mask_k = (
		src_mask
			.unsqueeze(1)
			.expand(-1, k, -1)
			.reshape(-1, src_mask.size(1))
	)

	return mem_k, mask_k
