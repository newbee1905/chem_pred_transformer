#!/bin/bash

source .venv/bin/activate

# CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v5.ckpt"
# CKPT_PATH="last-epoch-checkpoint-finetune-uspto-sep-bart_small_v5.ckpt"
# CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v7-tanimoto-v3.ckpt"
# CKPT_PATH="best-checkpoint-finetune-uspto-sep-chemformer_small_v8-tanimoto-v1.ckpt"
# CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
# CKPT_PATH="best-checkpoint-finetune-private-rotate-bart_small.ckpt"
# CKPT_PATH="best-checkpoint-finetune-private-bart_small-v8.ckpt"
CKPT_PATH="best-checkpoint-finetune-private-bart_small-tanimoto-v31.ckpt"
# CKPT_PATH="last-epoch-checkpoint-finetune-private-bart_small-v9.ckpt"
MODEL="bart_small"
PRETRAIN_STATE_DICT="bart_uspto_finetuned_v8-v8.pth"

	# +ckpt_path=train_checkpoints/${CKPT_PATH} \
	# +ckpt_path=train_checkpoints_server/${CKPT_PATH} \

	# dataset=uspto \
	# +dataset.mode=sep \

	# dataset=private_rotate \

python main.py \
	+ckpt_path=train_checkpoints/${CKPT_PATH} \
	num_workers=16 \
	task=test \
	model=$MODEL \
	dataset=private \
	+module.sampler=beam \
	+module.beam_size=10 \
	batch_size=4
