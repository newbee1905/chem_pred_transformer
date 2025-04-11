import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def greedy_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor = None,
	max_length: int = 50, start_token_id: int = 0,
	end_token_id: int = 1,
	kv_cache: bool = False,
	length_penalty_alpha: float = 0.8,
) -> torch.Tensor:
	"""Greedy decoding sampler."""

	device = memory.device
	bsz = memory.size(1)

	generated = torch.full(
		(bsz, 1), start_token_id, dtype=torch.long,
		device=device
	)
	finished = torch.zeros(bsz, dtype=torch.bool, device=device)

	for t in range(max_length - 1):
		write_idx = torch.arange(t + 1, device=device) if kv_cache else None
		start_pos = t if kv_cache else 0
		dec = model.decode(
			generated,
			memory,
			tgt_mask=None,
			memory_mask=src_mask,
			kv_write_indices=write_idx,
			start_pos=t,
		)

		last_dec = dec[-1, :, :] if kv_cache else dec[-1]

		logits = model.token_fc(last_dec)

		if length_penalty_alpha > 0:
			lp = ((5 + (t + 1)) / 6) ** length_penalty_alpha
			logits[..., end_token_id] -= math.log(lp)

		log_probs = F.log_softmax(logits, dim=-1)
		next_token = torch.argmax(log_probs, dim=-1, keepdim=True)

		next_token = torch.where(
			finished.unsqueeze(-1),
			torch.tensor(end_token_id, device=device),
			next_token,
		)

		generated = torch.cat([generated, next_token], dim=1)

		finished |= (next_token.squeeze(-1) == end_token_id)
		if finished.all():
			break

	return generated

def beam_search_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor = None,
	max_length: int = 256, start_token_id: int = 0,
	end_token_id: int = 1, beam_size: int = 5,
	length_penalty_alpha: float = 0.8, kv_cache: bool = False
) -> torch.Tensor:

	device = memory.device
	bsz = memory.size(1)
	vocab_size = model.token_fc.out_features

	memory = memory.repeat(1, beam_size, 1)
	if src_mask is not None:
		src_mask = src_mask.repeat_interleave(beam_size, dim=0)

	sequences = torch.full((bsz * beam_size, 1), start_token_id, dtype=torch.long, device=device)
	beam_scores = torch.zeros(bsz * beam_size, device=device)

	beam_scores[torch.arange(bsz * beam_size) % beam_size != 0] = -float("inf")

	finished = torch.zeros(bsz * beam_size, dtype=torch.bool, device=device)

	sequences = sequences.view(bsz, beam_size, -1)
	beam_scores = beam_scores.view(bsz, beam_size)
	finished = finished.view(bsz, beam_size)

	for t in range(max_length - 1):
		write_idx = torch.arange(t + 1, device=device) if kv_cache else None
		start_pos = t if kv_cache else 0

		flat_sequences = sequences.view(bsz * beam_size, -1)
		dec = model.decode(
			flat_sequences, memory, tgt_mask=None, memory_mask=src_mask,
			kv_write_indices=write_idx, start_pos=start_pos
		)

		last_dec = dec[-1, :, :] if kv_cache else dec[-1]
		# [bsz * beam_size, vocab_size]
		logits = model.token_fc(last_dec)

		if length_penalty_alpha > 0:
			lp = ((5 + t + 1) / 6) ** length_penalty_alpha
			logits[:, end_token_id] -= math.log(lp)

		log_probs = F.log_softmax(logits, dim=-1)

		# For beams already finished, force their distribution to only allow the EOS token.
		finished_flat = finished.view(bsz * beam_size)
		log_probs = log_probs.masked_fill(finished_flat.unsqueeze(1), -float("inf"))
		log_probs[finished_flat, end_token_id] = 0.0

		candidate_scores = beam_scores.view(bsz * beam_size, 1) + log_probs
		candidate_scores = candidate_scores.view(bsz, beam_size * vocab_size)

		top_scores, top_indices = torch.topk(candidate_scores, beam_size, dim=1)

		# [bsz, beam_size]
		beam_indices = top_indices // vocab_size	
		token_indices = top_indices % vocab_size	

		new_sequences = torch.gather(
			sequences, 1,
			beam_indices.unsqueeze(-1).expand(bsz, beam_size, t + 1),
		)
		new_sequences = torch.cat([new_sequences, token_indices.unsqueeze(-1)], dim=-1)

		new_finished = torch.gather(finished, 1, beam_indices) | (token_indices == end_token_id)

		sequences = new_sequences
		beam_scores = top_scores
		finished = new_finished

		if finished.any(dim=1).all():
			break

	sorted_scores, sort_indices = torch.sort(beam_scores, dim=1, descending=True)
	sorted_sequences = torch.gather(sequences, 1, sort_indices.unsqueeze(-1).expand_as(sequences))

	return sorted_sequences, sorted_scores
