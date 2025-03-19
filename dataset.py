import os
import gzip
import shutil
import requests
import logging
import zlib
from rdkit import Chem
from joblib import Parallel, delayed
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, random_split
import torch
from transformers import PreTrainedTokenizerFast

def process_molecule(mol):
	"""Convert a molecule to SMILES format, handling errors."""
	if mol is None:
		return None
	try:
		return Chem.MolToSmiles(mol, canonical=True)
	except Exception:
		return None


def download_file(url: str, save_path: str) -> None:
	logging.info(f"Downloading file from {url} ...")

	with requests.get(url, stream=True) as r:
		r.raise_for_status()

		with open(save_path, "wb") as f:
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)

	logging.info(f"Downloaded: {save_path}")


def decompress_gzip(sdf_gz_path: str, temp_sdf: str) -> bool:
	try:
		logging.info(f"Decompressing {sdf_gz_path} ...")

		with gzip.open(sdf_gz_path, "rb") as f_in, open(temp_sdf, "wb") as f_out:
			shutil.copyfileobj(f_in, f_out)

		logging.info("Decompression using gzip successful.")
		return True
	except EOFError:
		logging.warning("gzip decompression failed. Trying zlib.")
	
	try:
		with open(sdf_gz_path, "rb") as f:
			decompressed_data = zlib.decompress(f.read(), zlib.MAX_WBITS | 16)

		with open(temp_sdf, "wb") as f_out:
			f_out.write(decompressed_data)

		logging.info("Decompression using zlib successful.")
		return True
	except zlib.error as e:
		logging.error(f"Decompression failed: {e}")
		return False

def extract_smiles_from_sdf(sdf_gz_path: str, smiles_output_file: str, n_jobs: int = -1) -> None:
	temp_sdf = "temp.sdf"

	if not decompress_gzip(sdf_gz_path, temp_sdf):
		logging.error("Failed to decompress the file. Exiting function.")
		return

	if not os.path.exists(temp_sdf) or os.path.getsize(temp_sdf) == 0:
		logging.error("Decompressed file is empty. Possible corruption.")
		return

	logging.info("Checking first few lines of the decompressed file:")
	with open(temp_sdf, "r") as f:
		for _ in range(5):
			logging.info(f.readline().strip())

	logging.info("Parsing SDF file to extract SMILES ...")
	suppl = Chem.SDMolSupplier(temp_sdf, removeHs=True)
	smiles_list = []

	smiles_list = []
	with tqdm(total=len(suppl), desc="Extracting SMILES", unit="mol") as pbar:
		smiles_list = Parallel(n_jobs=n_jobs)(
			delayed(process_molecule)(mol) for mol in suppl
		)

		for _ in smiles_list:
			pbar.update(1)

	smiles_list = [smi for smi in smiles_list if smi]

	if not smiles_list:
		logging.warning("No valid SMILES extracted. Check the SDF file format.")

	with open(smiles_output_file, "w", encoding="utf-8") as f:
		for smi in smiles_list:
			f.write(smi + "\n")

	logging.info(f"Extracted {len(smiles_list)} SMILES to {smiles_output_file}")

	os.remove(temp_sdf)

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

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	sdf_url = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35.sdf.gz"
	sdf_file = "chembl_35.sdf.gz"
	smiles_file = "chembl_35.smi"

	if not os.path.exists(sdf_file):
		download_file(sdf_url, sdf_file)
	else:
		logging.info(f"{sdf_file} already exists.")

	if not os.path.exists(smiles_file):
		extract_smiles_from_sdf(sdf_file, smiles_file, n_jobs=4)
	else:
		logging.info(f"{smiles_file} already exists.")
