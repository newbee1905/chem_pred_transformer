import torch
import joblib
from tqdm import tqdm
import numpy as np

from models.bart import BART
from models.sampler import nucleus_sampler
from tokenisers.neochem import ChemformerTokenizerFast
from dataset.uspto import USPTODataset
from metrics import compute_batch_tanimoto_rewards
from torch.utils.data import DataLoader

CKPT_PATH = "train_checkpoints/best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
SOURCE_DATA_FILE = "USPTO_MIT.csv"
OUTPUT_FILE = "reward_data_with_hidden_states.joblib"
GENERATIONS_PER_REACTANT = 5
BATCH_SIZE = 16

print("Loading dataset and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = ChemformerTokenizerFast("bart_vocab.json")
vocab_size = tokenizer.vocab_size

max_length = collator.max_length

collator = BARTDataCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

ds = USPTODataset(USPTO_CSV_FILE, tokenizer, mode="sep")
dl = DataLoader(ds, batch_size=BATCH_SIZE)

print(f"Loaded {len(source_dataset)} source reactants.")


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

all_data = []
for batch in tqdm(source_loader, desc="Generating reward data"):
	src_tokens = batch['input_ids'].to(device)
	src_mask = batch['attention_mask'].to(device).eq(0)
	tgt_tokens = batch['labels'] 

	expanded_src_tokens = src_tokens.repeat_interleave(GENERATIONS_PER_REACTANT, dim=0)
	expanded_src_mask = src_mask.repeat_interleave(GENERATIONS_PER_REACTANT, dim=0)

	with torch.no_grad():
		generated_tokens = actor.generate(
			src=expanded_src_tokens,
			src_mask=expanded_src_mask,
			sampler=nucleus_sampler,
			top_p=0.9
		) # Shape: [bsz * G, SeqLen_tgt]

		memory = actor.encode(expanded_src_tokens, expanded_src_mask)
		_, _, decoder_hidden_states = actor.evaluate_actions(
			memory=memory,
			src_mask=expanded_src_mask,
			tgt_tokens=generated_tokens,
			pad_token_id=tokenizer.pad_token_id
		) # Shape: [seq - 1, bsz * G, d_model]

	target_smiles_list = tokenizer.batch_decode(tgt_tokens_for_reward.tolist(), skip_special_tokens=True)
	generated_smiles_list = tokenizer.batch_decode(generated_tokens.tolist(), skip_special_tokens=True)
	expanded_targets_str = [t for t in target_smiles_list for _ in range(GENERATIONS_PER_REACTANT)]
	scores = compute_batch_tanimoto_rewards(generated_smiles_list, expanded_targets_str, device=device)

	# Transpose hidden states to be [bsz, seq, d_model] 
	decoder_hidden_states = decoder_hidden_states.permute(1, 0, 2)

	src_tokens_np = expanded_src_tokens.cpu().numpy()
	product_tokens_np = generated_tokens.cpu().numpy()
	scores_np = scores.cpu().numpy()
	hidden_states_np = decoder_hidden_states.cpu().numpy()

	for i in range(len(scores_np)):
		all_data.append({
			"reactant_tokens": src_tokens_np[i],
			"product_tokens": product_tokens_np[i],
			"score": scores_np[i],
			"decoder_hidden_states": hidden_states_np[i]
		})

print(f"\nSaving {len(all_data)} data points to {OUTPUT_FILE}...")
joblib.dump(all_data, OUTPUT_FILE, compress=3) # Use compression to save space
print("Data generation and saving complete.")
