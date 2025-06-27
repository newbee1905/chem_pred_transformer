import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def greedy_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor = None,
	max_length: int = 50, start_token_id: int = 0,
	end_token_id: int = 1,
	kv_cache: bool = False,
	return_logpi: bool = False,
) -> torch.Tensor:
	"""Greedy decoding sampler."""

	device = memory.device
	bsz = memory.size(1)

	generated = torch.full(
		(bsz, 1), start_token_id, dtype=torch.long,
		device=device
	)
	finished = torch.zeros(bsz, dtype=torch.bool, device=device)

	if return_logpi:
		log_pi = torch.zeros(bsz, device=device)

	for t in range(max_length - 1):
		start_pos = t if kv_cache else 0
		input_ids = generated[:, -1:] if kv_cache else generated

		dec = model.decode(
			input_ids,
			memory,
			tgt_mask=None,
			memory_mask=src_mask,
			kv_cache=kv_cache,
			start_pos=start_pos,
		)

		last_dec = dec[-1, :, :] if kv_cache else dec[-1]

		logits = model.token_fc(last_dec)

		log_probs = F.log_softmax(logits, dim=-1)
		next_token = torch.argmax(log_probs, dim=-1, keepdim=True)

		next_token = torch.where(
			finished.unsqueeze(-1),
			torch.tensor(end_token_id, device=device),
			next_token,
		)

		if return_logpi:
			log_prob_t = log_probs.gather(1, next_token).squeeze(1)
			log_pi += torch.where(finished, 0.0, log_prob_t)

		generated = torch.cat([generated, next_token], dim=1)

		finished |= (next_token.squeeze(-1) == end_token_id)
		if finished.all():
			break

	if not return_logpi:
		return generated

	return generated, log_pi

def beam_search_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor = None,
	max_length: int = 256, start_token_id: int = 0,
	end_token_id: int = 1, beam_size: int = 5,
	length_penalty_alpha: float = 1, 
	kv_cache: bool = False,
) -> torch.Tensor:

	device = memory.device
	bsz = memory.size(1)
	vocab_size = model.token_fc.out_features

	# [seq_len, bsz, dim] -> [seq_len, bsz, beam, dim]
	memory = memory.unsqueeze(2).repeat(1, 1, beam_size, 1)
	memory = memory.view(memory.size(0), bsz * beam_size, -1)
	if src_mask is not None:
		src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1)
		src_mask = src_mask.view(bsz * beam_size, -1)

	sequences = torch.full((bsz * beam_size, 1), start_token_id, dtype=torch.long, device=device)
	beam_scores = torch.zeros(bsz * beam_size, device=device)
	finished = torch.zeros(bsz * beam_size, dtype=torch.bool, device=device)

	sequences = sequences.view(bsz, beam_size, -1)
	beam_scores = beam_scores.view(bsz, beam_size)
	beam_scores[:, 1:] = -float('inf')
	finished = finished.view(bsz, beam_size)

	for t in range(max_length - 1):
		start_pos = t if kv_cache else 0

		if kv_cache:
			flat_sequences = sequences[:, :, -1:].reshape(bsz * beam_size, 1)
		else:
			flat_sequences = sequences.view(bsz * beam_size, -1)

		dec = model.decode(
			flat_sequences, memory, tgt_mask=None, memory_mask=src_mask,
			kv_cache=kv_cache, start_pos=start_pos
		)

		last_dec = dec[-1, :, :] if kv_cache else dec[-1]
		# [bsz * beam_size, vocab_size]
		logits = model.token_fc(last_dec)
		log_probs = F.log_softmax(logits, dim=-1)

		finished_flat = finished.view(-1)
		log_probs.masked_fill_(
			finished_flat.unsqueeze(1), -float('inf')
		)
		log_probs[finished_flat, end_token_id] = 0.0

		flat_scores = beam_scores.view(bsz * beam_size, 1) + log_probs
		candidate_scores = flat_scores.view(bsz, beam_size * vocab_size)

		top_scores, top_indices = torch.topk(candidate_scores, beam_size, dim=1)

		# [bsz, beam_size]
		beam_indices = top_indices // vocab_size	
		token_indices = top_indices % vocab_size	

		prev_sequences = torch.gather(
			sequences, 1,
			beam_indices.unsqueeze(-1).expand(bsz, beam_size, t + 1),
		)
		sequences = torch.cat([prev_sequences, token_indices.unsqueeze(-1)], dim=-1)

		finished = torch.gather(finished, 1, beam_indices) | (token_indices == end_token_id)
		beam_scores = top_scores

		if finished.all():
			break

	seq_len = sequences.size(-1)
	idxs = torch.arange(seq_len, device=device)
	idxs = idxs.view(1, 1, -1)
	eos_positions = torch.where(
		sequences == end_token_id,
		idxs,
		torch.full_like(idxs, seq_len)
	)
	lengths = eos_positions.min(dim=-1).values + 1

	"""
	Based on Chemformer normalise method

	Normalise log-likelihoods using the length of the constructed sequence

	Equation from:
	Wu, Yonghui, et al.
	"Google's neural machine translation system: Bridging the gap between human and machine translation."
	arXiv preprint arXiv:1609.08144 (2016).
	"""
	if length_penalty_alpha and length_penalty_alpha > 0:
		penalty = torch.pow((5.0 + lengths), length_penalty_alpha) / (6.0 ** length_penalty_alpha)
		norm_scores = beam_scores / penalty
	else:
		norm_scores = beam_scores

	sorted_scores, sort_indices = torch.sort(norm_scores, dim=1, descending=True)
	sorted_sequences = torch.gather(sequences, 1, sort_indices.unsqueeze(-1).expand_as(sequences))

	return sorted_sequences, sorted_scores

def nucleus_sampler(
	model,
	memory: torch.Tensor,
	src_mask: torch.Tensor = None,
	max_length: int = 50,
	start_token_id: int = 0,
	end_token_id: int = 1,
	kv_cache: bool = False,
	return_logpi: bool = False,
	top_p: float = 0.9,
	temperature: float = 1.0,
) -> torch.Tensor:
	"""Nucleus (top-p) sampling: dynamic cutoff by cumulative probability."""
	device = memory.device
	bsz = memory.size(1)
	generated = torch.full((bsz, 1), start_token_id, device=device, dtype=torch.long)
	finished = torch.zeros(bsz, dtype=torch.bool, device=device)
	if return_logpi:
		log_pi = torch.zeros(bsz, device=device)

	for t in range(max_length - 1):
		start_pos = t if kv_cache else 0
		input_ids = generated[:, -1:] if kv_cache else generated

		dec = model.decode(
			input_ids,
			memory,
			tgt_mask=None,
			memory_mask=src_mask,
			kv_cache=kv_cache,
			start_pos=start_pos,
		)

		last_dec = dec[-1] if not kv_cache else dec[-1, :, :]
		logits = model.token_fc(last_dec) / temperature

		# sort logits descending, compute softmax probabilities
		sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
		cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# keep tokens up to top_p
		cutoff = cum_probs > top_p
		cutoff[..., 1:] &= ~cutoff[..., :-1]
		keep = ~cutoff
		# mask tokens beyond nucleus
		mask = torch.full_like(logits, float('-inf'))
		mask.scatter_(1, sorted_idx, sorted_logits.masked_fill(~keep, float('-inf')))
		probs = F.softmax(mask, dim=-1)

		next_token = torch.multinomial(probs, num_samples=1)
		if return_logpi:
			log_prob_t = torch.log(probs.gather(1, next_token).squeeze(1) + 1e-9)
			log_pi += torch.where(finished, 0.0, log_prob_t)

		next_token = torch.where(
			finished.unsqueeze(-1),
			torch.tensor(end_token_id, device=device),
			next_token,
		)

		generated = torch.cat([generated, next_token], dim=1)
		finished |= (next_token.squeeze(-1) == end_token_id)
		if finished.all():
			break

	if not return_logpi:
		return generated

	return generated, log_pi
