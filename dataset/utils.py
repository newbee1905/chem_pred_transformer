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
