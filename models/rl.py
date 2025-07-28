import re

from models.mha import KVCacheMHA
from models.utils import AttentionPooler

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl
from concurrent.futures import ThreadPoolExecutor

VALID_SMILES_CHARS_RE = re.compile(r'[^\w\(\)\[\]=\-\#/\+\.@%0-9]')
INVALID_BOND_SEQUENCE_RE = re.compile(r'==|##|==#|##=')
ATOM_PATTERN_RE = re.compile(r'[A-Z][a-z]?')

def _intermediate_smiles_reward(
	partial_smiles: str,
	reactant_smiles: str
) -> float:
	"""
	Performs syntax checks and atom-count matching for a partial SMILES.
	"""
	# Syntax heuristics
	if VALID_SMILES_CHARS_RE.search(partial_smiles):
		return 0.0
	if (partial_smiles.count('(') < partial_smiles.count(')') or
			partial_smiles.count('[') < partial_smiles.count(']')):
		return 0.0
	if INVALID_BOND_SEQUENCE_RE.search(partial_smiles):
		return 0.0

	# Atom counts
	reac_atoms = ATOM_PATTERN_RE.findall(reactant_smiles)
	if not reac_atoms:
		return 1.0

	prod_atoms = ATOM_PATTERN_RE.findall(partial_smiles)

	diff = abs(len(prod_atoms) - len(reac_atoms))
	return max(0.0, 1.0 - diff / len(reac_atoms))

def intermediate_smiles_reward(
	partial_smiles_list: list[str],
	reactant_smiles_list: list[str]
) -> list[float]:

	with ThreadPoolExecutor() as executor:
		args = zip(partial_smiles_list, reactant_smiles_list)
		scores = list(executor.map(lambda x: _intermediate_smiles_reward(*x), args))

	return scores

class Critic(nn.Module):
	def __init__(self, d_model: int, n_heads: int = 8, is_per_step: bool = False):
		super().__init__()
		self.d_model = d_model
		self.n_heads = n_heads
		self.is_per_step = is_per_step

		self.cross_attn = KVCacheMHA(
			d_model=self.d_model,
			n_heads=self.n_heads
		)

		self.pooler = AttentionPooler(self.d_model)

		self.value_head = nn.Sequential(
			nn.Linear(self.d_model, self.d_model // 2),
			nn.SiLU(),
			nn.Linear(self.d_model // 2, 1)
		)

	def forward(self, decoder_hidden_states, memory, src_mask=None):
		attn_output = self.cross_attn(
			query=decoder_hidden_states,
			key=memory,
			value=memory,
			attn_mask=src_mask,
			is_causal=False,
			kv_cache=False,
		)

		if not self.is_per_step:
			pooled_memory = self.pooler(attn_output)
			value = self.value_head(pooled_memory).squeeze(-1)
		else:
			value = self.value_head(attn_output).squeeze(-1)

		return value
	# def forward(self, memory, src_mask=None):
	# 	pooled_memory = self.pooler(memory, src_mask)
	# 	value = self.value_head(pooled_memory).squeeze(-1)

	# 	return value
