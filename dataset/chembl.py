from rdkit import Chem

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerFast

# from dataset.utils import download_file, extract_smiles_from_sdf

import logging

class ChemBL35Dataset(Dataset):
	def __init__(
		self, smiles_file: str, tokenizer: PreTrainedTokenizerFast,
		max_length: int = 64, noise_prob: float = 0.25
	):
		with open(smiles_file, "r", encoding="utf-8") as f:
			self.smiles_list = [line.strip() for line in f if line.strip()]

		self.tokenizer = tokenizer
		self.max_length = max_length
		self.noise_prob = noise_prob

	def __len__(self):
		return len(self.smiles_list)

	def __getitem__(self, idx):
		original_smiles = self.smiles_list[idx]

		mol = Chem.MolFromSmiles(original_smiles)
		if mol is None:
			input_smiles = original_smiles
			label_smiles = original_smiles
		else:
			input_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=True)
			label_smiles = Chem.MolToSmiles(mol, canonical=True)

		if self.tokenizer.bos_token is not None:
			if not input_smiles.startswith(self.tokenizer.bos_token):
				input_smiles = self.tokenizer.bos_token + input_smiles
			if not label_smiles.startswith(self.tokenizer.bos_token):
				label_smiles = self.tokenizer.bos_token + label_smiles

		if self.tokenizer.eos_token is not None:
			if not input_smiles.endswith(self.tokenizer.eos_token):
				input_smiles = input_smiles + self.tokenizer.eos_token
			if not label_smiles.endswith(self.tokenizer.eos_token):
				label_smiles = label_smiles + self.tokenizer.eos_token

		enc_input = self.tokenizer(
			input_smiles,
			truncation=True,
			max_length=self.max_length,
			padding="max_length",
			return_tensors="pt",
		)
		enc_label = self.tokenizer(
			label_smiles,
			truncation=True,
			max_length=self.max_length,
			padding="max_length",
			return_tensors="pt",
		)

		input_ids = enc_input["input_ids"].squeeze(0)
		attention_mask = enc_input["attention_mask"].squeeze(0) 

		non_pad_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]
		num_to_mask = max(1, int(len(non_pad_indices) * self.noise_prob))

		mask_indices = torch.randperm(len(non_pad_indices))[:num_to_mask]
		selected_mask_positions = non_pad_indices[mask_indices]

		noisy_input_ids = input_ids.clone()
		noisy_input_ids[selected_mask_positions] = self.tokenizer.mask_token_id

		return {
			"input_ids": noisy_input_ids,
			"attention_mask": attention_mask,
			"labels": enc_label["input_ids"].squeeze(0)
		}

# if __name__ == "__main__":
# 	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# 	sdf_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35.sdf.gz"
# 	sdf_file = "chembl_35.sdf.gz"
# 	smiles_file = "chembl_35.smi"
#
# 	if not os.path.exists(sdf_file):
# 		download_file(sdf_url, sdf_file)
# 	else:
# 		logging.info(f"{sdf_file} already exists.")
#
# 	if not os.path.exists(smiles_file):
# 		extract_smiles_from_sdf(sdf_file, smiles_file, n_jobs=4)
# 	else:
# 		logging.info(f"{smiles_file} already exists.")
