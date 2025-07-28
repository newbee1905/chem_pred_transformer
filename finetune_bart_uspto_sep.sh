#!/bin/bash

source .venv/bin/activate

MODEL="bart_small"
TASK="finetune"
# PRETRAIN_STATE_DICT="best-checkpoint-pretrain-zinc-${MODEL}_e5_v0.pth"
# PRETRAIN_STATE_DICT="best-exact-ppo.pth"
# PRETRAIN_STATE_DICT="bart_uspto_finetuned_v8-v8.pth"
PRETRAIN_STATE_DICT="bart_uspto_finetuned_v8-tanimoto-v8.pth"
NAME="private-${MODEL}"

# dataset=uspto +dataset.mode=sep task=train task_type=finetune \
# +pretrained_state_dict="best-checkpoint-$PRETRAIN_STATE_DICT" \
# +ckpt_path="train_checkpoints/best-checkpoint-finetune-uspto-sep-${MODEL}_small_v0.ckpt" \
# +ckpt_path="train_checkpoints/best-checkpoint-finetune-uspto-sep-${MODEL}_aux_3-v1.ckpt" \
# pth_path="best-checkpoint-$TASK-$NAME.pth" \
# pth_path="best-exact-ppo.pth" \
# pth_path="bart_uspto_finetuned_v8-v8.pth" \

#	+model.aux_head=true +dataset.collator.aux_head=true \
#	+dataset.collator.aux_prop_stats_path=aux_prop_stats.pickle \

HYDRA_FULL_ERROR=1 python main.py model="$MODEL" \
	+pretrained_state_dict="$PRETRAIN_STATE_DICT" \
	dataset=private task=train task_type=finetune \
	callbacks.1.filename="best-checkpoint-$TASK-$NAME" \
	callbacks.2.filename="best-checkpoint-$TASK-$NAME-tanimoto" \
	callbacks.3.filename="last-epoch-checkpoint-$TASK-$NAME" \
	loggers.0.name="$TASK-$NAME" \
	+trainer.max_epochs=100 \
	num_workers=16 \
	+kv_cache=true
