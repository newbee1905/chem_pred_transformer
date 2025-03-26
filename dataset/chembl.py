from rdkit import Chem

from transformers import PreTrainedTokenizerFast
from tokenisers.chemformer import ChemformerTokenizer

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from dataset.pretrain import PretrainBARTDataset

class ChemBL35Dataset(PretrainBARTDataset):
	"""ChemBL pretraining dataset that reads SMILES strings from a file."""
	
	def __init__(self,
		smiles_file: str,
		tokenizer: PreTrainedTokenizerFast | ChemformerTokenizer,
		max_length: int = 64,
		noise_prob: float = 0.5,
		span_lambda: float = 3,
		tokenizer_type: str = "hf"
	):
		with open(smiles_file, "r", encoding="utf-8") as f:
			self.smiles_list = [line.strip() for line in f if line.strip()]
		self.tokenizer_type = tokenizer_type

		if tokenizer_type == "hf":
			self.vocab_size = tokenizer.vocab_size
		elif tokenizer_type == "chemformer":
			self.vocab_size = len(tokenizer)
		else:
			raise ValueError("Invalid tokenizer_type. Use 'hf' or 'chemformer'.")

		super().__init__(tokenizer, max_length, noise_prob, span_lambda)
