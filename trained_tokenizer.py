import os
import sys
import requests
import logging

from tokenizers.neocart import SMILESTokenizer

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	smiles_file = "chembl_35.smi"

	tokenizer = SMILESTokenizer.train_new(
		files=[dataset_file],
		vocab_size=vocab_size,
		min_frequency=min_frequency
	)

	logging.info(f"Tokenizer trained. Vocabulary size: {tokenizer.vocab_size}")

	save_dir = "trained_tokenizer"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	tokenizer.save_pretrained(save_dir)
	logging.info(f"Tokenizer vocabulary saved to directory: {save_dir}")
