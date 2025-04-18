import torch
from rdkit import Chem

from typing import List, Tuple

class BARTDataCollator:
	def __init__(self, tokenizer, max_length: int = 64):
		self.tokenizer = tokenizer
		self.max_length = max_length

		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.bos_token = tokenizer.bos_token
		self.eos_token = tokenizer.eos_token

	def __call__(self, batch: List[Tuple[str, str]]) -> dict[str, torch.Tensor]:
		inp_smiles, label_smiles = zip(*batch)

		enc = self.tokenizer(
			inp_smiles,
			text_target=label_smiles,
			truncation=True,
			max_length=self.max_length,
			padding="max_length",
			return_tensors="pt",
		)

		enc["decoder_attention_mask"] = (enc["labels"] != self.pad_token_id).long()

		return {
			"input_ids": enc["input_ids"],
			"attention_mask": enc["attention_mask"],
			"labels": enc["labels"],
			"labels_attention_mask": enc["decoder_attention_mask"],
		}
