#!/bin/bash

source .venv/bin/activate

# CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v5.ckpt"
# CKPT_PATH="last-epoch-checkpoint-finetune-uspto-sep-bart_small_v5.ckpt"
# CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v7-tanimoto-v3.ckpt"
# CKPT_PATH="best-checkpoint-finetune-uspto-sep-chemformer_small_v8-tanimoto-v1.ckpt"
CKPT_PATH="best-checkpoint-finetune-uspto-sep-bart_small_v8-v8.ckpt"
MODEL="bart_small"

	# +ckpt_path=train_checkpoints/${CKPT_PATH} \

python main.py \
	num_workers=16 \
	task=test \
	+ckpt_path=train_checkpoints_server/${CKPT_PATH} \
	dataset=uspto \
	+dataset.mode=sep \
	model=$MODEL \
	+module.sampler=beam \
	+module.beam_size=20 \
	batch_size=4
