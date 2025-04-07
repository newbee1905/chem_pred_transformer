import os
import json
import tempfile
from typing import List, Optional, Union

from transformers import PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration
from tokenizers import Tokenizer, decoders, pre_tokenizers

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import Strip, Replace

from rdkit import Chem

SMI_REGEX_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|@|\?|>>|\*|\%[0-9]{2,3}|[0-9]+)"

class SMILESTokenizer(PreTrainedTokenizerFast):
	vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

	def __init__(
		self,
		vocab_file: Optional[str] = None,
		merges_file: Optional[str] = None,
		tokenizer_file: Optional[str] = None,
		unk_token: str = "<unk>",
		bos_token: str = "<s>",
		eos_token: str = "</s>",
		pad_token: str = "<pad>",
		mask_token: str = "<mask>",
		**kwargs,
	):
		if tokenizer_file is not None:
			super().__init__(
				tokenizer_file=tokenizer_file,
				unk_token=unk_token,
				bos_token=bos_token,
				eos_token=eos_token,
				pad_token=pad_token,
				mask_token=mask_token,
				**kwargs
			)
			return

		model = BPE(vocab_file, merges_file) if vocab_file and merges_file else BPE()
		tokenizer = Tokenizer(model)

		tokenizer.normalizer = normalizers.Sequence([
			normalizers.Strip(),
			normalizers.Replace("\n", ""),
		])

		tokenizer.pre_tokenizer = pre_tokenizers.Split(
			pattern=SMI_REGEX_PATTERN,
			behavior="merged_with_previous"
		)

		tokenizer.decoder = decoders.ByteLevel()

		super().__init__(
			tokenizer_object=tokenizer,
			unk_token=unk_token,
			bos_token=bos_token,
			eos_token=eos_token,
			pad_token=pad_token,
			mask_token=mask_token,
			**kwargs
		)

	@classmethod
	def from_custom_vocab(cls, vocab_file: str, **kwargs) -> "SMILESTokenizer":
		with open(vocab_file, 'r') as f:
			lines = f.read().splitlines()

		vocab, merges = {}, []
		token_id = 0

		for line in lines:
			parts = line.strip().split()
			if len(parts) == 2:
				merges.append(f"{parts[0]} {parts[1]}")

				for token in parts:
					if token not in vocab:
						vocab[token] = token_id
						token_id += 1

				merged = parts[0] + parts[1]
				if merged not in vocab:
					vocab[merged] = token_id
					token_id += 1

		with tempfile.TemporaryDirectory() as temp_dir:
			vocab_file_path = os.path.join(temp_dir, "vocab.json")
			merges_file_path = os.path.join(temp_dir, "merges.txt")

			with open(vocab_file_path, 'w') as vf:
				json.dump(vocab, vf)

			with open(merges_file_path, 'w') as mf:
				mf.writelines(f"{merge}\n" for merge in merges)

			return cls(vocab_file=vocab_file_path, merges_file=merges_file_path, **kwargs)

	# @classmethod
	# def train_new(
	# 	cls,
	# 	files: List[str],
	# 	vocab_size: int = 30000,
	# 	min_frequency: int = 2000,
	# 	special_tokens: Optional[List[str]] = None,
	# 	save_path: Optional[str] = None,
	# 	augment_factor: int = 1,
	# 	**kwargs,
	# ) -> "SMILESTokenizer":
	# 	default_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
	# 	if special_tokens:
	# 		default_special_tokens.extend(special_tokens)
	#
	# 	smiles_list = []
	# 	for file in files:
	# 		with open(file, "r") as f:
	# 			for line in f:
	# 				smi = line.strip()
	# 				if smi:
	# 					smiles_list.append(smi)
	#
	# 	if augment_factor > 0:
	# 		augmented = []
	# 		for smi in smiles_list:
	# 			augmented.append(smi)
	# 			try:
	# 				mol = Chem.MolFromSmiles(smi)
	# 				for _ in range(augment_factor):
	# 					rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
	# 					augmented.append(rand_smi)
	# 			except Exception as e:
	# 				continue
	# 		smiles_list = augmented
	#
	# 	tokenizer = Tokenizer(BPE())
	# 	tokenizer.pre_tokenizer = pre_tokenizers.Split(
	# 		pattern=SMI_REGEX_PATTERN,
	# 		behavior="merged_with_previous"
	# 	)
	#
	# 	trainer = BpeTrainer(
	# 		vocab_size=vocab_size,
	# 		min_frequency=min_frequency,
	# 		special_tokens=default_special_tokens
	# 	)
	#
	# 	tokenizer.train_from_iterator(smiles_list, trainer)
	# 	tokenizer.decoder = decoders.ByteLevel()
	#
	# 	# Handle saving
	# 	if save_path:
	# 		tokenizer.save(save_path)
	# 		return cls(tokenizer_file=save_path, **kwargs)
	#
	# 	with tempfile.TemporaryDirectory() as temp_dir:
	# 		tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
	# 		tokenizer.save(tokenizer_path)
	# 		return cls(tokenizer_file=tokenizer_path, **kwargs)
	@classmethod
	def train_new(
		cls,
		files: List[str],
		vocab_size: int = 30000,
		min_frequency: int = 2000,
		special_tokens: Optional[List[str]] = None,
		save_path: Optional[str] = None,
		augment_factor: int = 2,
		batch_size: int = 1000,
		**kwargs,
	) -> "SMILESTokenizer":
		default_special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
		if special_tokens:
			default_special_tokens.extend(special_tokens)

		tokenizer = Tokenizer(BPE())
		tokenizer.pre_tokenizer = pre_tokenizers.Split(
			pattern=SMI_REGEX_PATTERN,
			behavior="merged_with_previous"
		)

		trainer = BpeTrainer(
			vocab_size=vocab_size,
			min_frequency=min_frequency,
			special_tokens=default_special_tokens
		)

		def batched_iterator():
			for file in files:
				with open(file, "r") as f:
					batch = []
					for line in f:
						smi = line.strip()
						if not smi:
							continue
							
						for part in smi.split("."):
							part = part.strip()
							if part:
								batch.append(part)
								
								if len(batch) >= batch_size:
									for item in batch:
										yield item
									batch.clear()
									
					for item in batch:
						yield item

		def memory_efficient_augmentation():
			for smi in batched_iterator():
				yield smi
				if augment_factor <= 0:
					continue
					
				try:
					mol = Chem.MolFromSmiles(smi)
					if mol is None:
						continue
						
					for _ in range(augment_factor):
						try:
							rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
							for part in rand_smi.split("."):
								part = part.strip()
								if part:
									yield part
						except Exception:
							continue
				except Exception:
					continue

		tokenizer.train_from_iterator(memory_efficient_augmentation(), trainer)
		tokenizer.decoder = decoders.ByteLevel()

		if save_path:
			tokenizer.save(save_path)
			return cls(tokenizer_file=save_path, **kwargs)

		with tempfile.TemporaryDirectory() as temp_dir:
			tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
			tokenizer.save(tokenizer_path)
			return cls(tokenizer_file=tokenizer_path, **kwargs)

# Example usage:
if __name__ == "__main__":
	tokenizer = SMILESTokenizer.from_pretrained("./trained_tokenizer")

	smi = "[H][C@@]1(C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CCCN=C(N)N)C(N)=O)CCCN1C(=O)[C@]1([H])CCCN1C(=O)CNC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)CNC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1c[nH]cn1)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)CNC(=O)[C@@]([H])(NC(=O)[C@H](CC(C)C)NC(=O)[C@@]([H])(NC(=O)[C@@H](N)CCSC)[C@@H](C)O)[C@@H](C)CC"
	print(smi)
	batch = tokenizer([smi], return_tensors="pt",  max_length=256, padding="max_length")
	print(batch)
	decode_batch = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
	print(decode_batch[0].split())

	for _ in range(10):
		mol = Chem.MolFromSmiles(smi)
		print(Chem.MolToSmiles(mol, doRandom=True, canonical=True))
