from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import joblib
from tqdm.auto import tqdm
import numpy as np
import lmdb
import pickle

from trainer.bart import BARTModel
from models.bart import BART
from models.sampler import beam_search_sampler
from tokenisers.neochem import ChemformerTokenizerFast
from dataset.uspto import USPTODataset
from dataset.base import BARTDataCollator
from metrics import compute_batch_tanimoto_rewards
from torch.utils.data import DataLoader

CKPT_PATH = "train_checkpoints/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
USPTO_CSV_FILE = "USPTO_MIT.csv"
OUTPUT_DB_PATH = "reward_data.lmdb" 
GENERATIONS_PER_REACTANT = 1
BATCH_SIZE = 256
LMDB_MAP_SIZE = 1024**4

print("Loading dataset and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ChemformerTokenizerFast("bart_vocab.json")
vocab_size = tokenizer.vocab_size

MAX_LENGTH = 282

collator = BARTDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

ds = USPTODataset(USPTO_CSV_FILE, tokenizer, mode="sep")
dl = DataLoader(
	ds,
	batch_size=BATCH_SIZE,
	shuffle=True,
	num_workers=10,
	collate_fn=collator,
)

max_length = collator.max_length
print(f"Loaded {len(ds)} source reactants.")

# --- Model Loading ---
MODEL_CONFIG = {
	"vocab_size": vocab_size,
	"d_model": 512,
	"n_layers": 6,
	"n_heads": 8,
	"d_ff": 2048,
	"dropout": 0.1,
	"max_seq_len": max_length,
	"aux_head": False,
}

untrained_bart_nn_module = BART(**MODEL_CONFIG)

print("Loading trained LightningModule from checkpoint...")
lightning_model = BARTModel.load_from_checkpoint(
	checkpoint_path=CKPT_PATH,
	model=untrained_bart_nn_module,
	tokenizer=tokenizer
)
print("Model loaded successfully.")

actor = lightning_model.model
actor = actor.eval()

print(f"Setting up LMDB database at {OUTPUT_DB_PATH}...")
env = lmdb.open(OUTPUT_DB_PATH, map_size=LMDB_MAP_SIZE, writemap=True)
data_idx = 0

for batch in tqdm(dl, desc="Generating and saving reward data"):
	src_tokens = batch['input_ids'].to(device)
	src_mask = batch['attention_mask'].to(device).eq(0)
	tgt_tokens = batch['labels']

	expanded_src_tokens = src_tokens.repeat_interleave(GENERATIONS_PER_REACTANT, dim=0)
	expanded_src_mask = src_mask.repeat_interleave(GENERATIONS_PER_REACTANT, dim=0)

	with torch.no_grad():
		generated_tokens, _ = actor.generate(
			src=expanded_src_tokens,
			src_mask=expanded_src_mask,
			sampler=beam_search_sampler,
			beam_size=3
		)
		generated_tokens = generated_tokens[:, 0, :]

		memory = actor.encode(expanded_src_tokens, expanded_src_mask)
		_, _, decoder_hidden_states = actor.evaluate_actions(
			memory=memory,
			src_mask=expanded_src_mask,
			tgt_tokens=generated_tokens,
			pad_token_id=tokenizer.pad_token_id
		)

	target_smiles_list = tokenizer.batch_decode(tgt_tokens.tolist(), skip_special_tokens=True)
	generated_smiles_list = tokenizer.batch_decode(generated_tokens.tolist(), skip_special_tokens=True)
	expanded_targets_str = [t for t in target_smiles_list for _ in range(GENERATIONS_PER_REACTANT)]
	scores = compute_batch_tanimoto_rewards(generated_smiles_list, expanded_targets_str, device=device)

	decoder_hidden_states = decoder_hidden_states.permute(1, 0, 2)

	src_tokens_np = expanded_src_tokens.cpu().numpy()
	product_tokens_np = generated_tokens.cpu().numpy()
	scores_np = scores.cpu().numpy()
	hidden_states_np = decoder_hidden_states.cpu().numpy()

	with env.begin(write=True) as txn:
		for i in range(len(scores_np)):
			data_point = {
				"reactant_tokens": src_tokens_np[i],
				"product_tokens": product_tokens_np[i],
				"score": scores_np[i],
				"decoder_hidden_states": hidden_states_np[i]
			}
			value = pickle.dumps(data_point)
			key = str(data_idx).encode('utf-8')
			txn.put(key, value)
			data_idx += 1

with env.begin(write=True) as txn:
	txn.put(b'__len__', str(data_idx).encode('utf-8'))

print(f"\nSaved {data_idx} data points to {OUTPUT_DB_PATH}")
env.close()
print("Data generation and saving complete.")
