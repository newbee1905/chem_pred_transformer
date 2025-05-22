#!/bin/bash

source .venv/bin/activate

MODEL="bart_small"
TASK="finetune"
PRETRAIN_STATE_DICT="pretrain-zinc-${MODEL}_e5_v0.pth"
NAME="uspto-sep-${MODEL}_v9"

# +pretrained_state_dict="best-checkpoint-$PRETRAIN_STATE_DICT" \
# +ckpt_path="train_checkpoints/best-checkpoint-finetune-uspto-sep-${MODEL}_small_v0.ckpt" \

#	+model.aux_head=true +dataset.collator.aux_head=true \
#	+dataset.collator.aux_prop_stats_path=aux_prop_stats.pickle \

HYDRA_FULL_ERROR=1 python main.py model="$MODEL" \
	+pretrained_state_dict="best-checkpoint-$PRETRAIN_STATE_DICT" \
	dataset=uspto +dataset.mode=sep task=train task_type=finetune \
	+ckpt_path="train_checkpoints/best-checkpoint-finetune-uspto-sep-${MODEL}_aux_3-v1.ckpt" \
	pth_path="best-checkpoint-$TASK-$NAME.pth" \
	callbacks.1.filename="best-checkpoint-$TASK-$NAME" \
	callbacks.2.filename="best-checkpoint-$TASK-$NAME-tanimoto" \
	callbacks.3.filename="last-epoch-checkpoint-$TASK-$NAME" \
	loggers.0.name="$TASK-$NAME" \
	+trainer.max_epochs=100 \
	num_workers=16
	+kv_cache=true
