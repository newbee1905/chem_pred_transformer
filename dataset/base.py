import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski

from typing import List, Tuple, Optional

import pickle

class BARTDataCollator:
	scalar_props = [
		"MolWt", "LogP", "TPSA",
		"NumHDonors", "NumHAcceptors",
		"NumRotatableBonds", "RingCount"
	]

	prop_funcs = {
		"MolWt": Descriptors.MolWt,
		"LogP": Crippen.MolLogP,
		"TPSA": rdMolDescriptors.CalcTPSA,
		"NumHDonors": Lipinski.NumHDonors,
		"NumHAcceptors": Lipinski.NumHAcceptors,
		"NumRotatableBonds": Lipinski.NumRotatableBonds,
		"RingCount": rdMolDescriptors.CalcNumRings,
	}

	def __init__(self, tokenizer, max_length: int = 64, aux_head: bool = False, aux_prop_stats_path: Optional[str] = None):
		self.tokenizer = tokenizer
		self.max_length = max_length

		self.pad_token_id = tokenizer.pad_token_id
		self.mask_token_id = tokenizer.mask_token_id
		self.bos_token = tokenizer.bos_token
		self.eos_token = tokenizer.eos_token

		self.aux_head = aux_head
		self.aux_prop_stats: Optional[Dict] = None
		if self.aux_head and aux_prop_stats_path:
			try:
				with open(aux_prop_stats_path, "rb") as f:
					self.aux_prop_stats = pickle.load(f)
			except:
					self.aux_prop_stats = None

	def _normalize_value(self, value: float, name: str, eps: float = 1e-6) -> float:
		# Z-score normalisation 
		mean = self.aux_prop_stats[name]['mean']	
		std = self.aux_prop_stats[name]['std']	

		return (value - mean) / (std + eps)

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

		enc["labels_attention_mask"] = (enc["labels"] != self.pad_token_id).long()

		if self.aux_head:
			# removing eos and bos
			# TODO: find a better method to calculate properties of smiles
			react_mols = [Chem.MolFromSmiles(s[1:-1].replace(">", ".")) for s in inp_smiles]
			# prod_mols = [Chem.MolFromSmiles(s[1:-1]) for s in label_smiles]

			# for side, mols in (("react", react_mols), ("prod", prod_mols)):
			for side, mols in (("react", react_mols), ):
				for name, func in self.prop_funcs.items():
					vals = [self._normalize_value(func(m), name) if m is not None else 0 for m in mols]
					enc[f"aux_{side}_{name}"] = torch.tensor(vals, dtype=torch.float)

		return enc
