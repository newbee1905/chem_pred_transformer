import torch
import torch.nn as nn
import torch.nn.functional as F

def greedy_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor,
	max_length: int = 50, start_token_id: int = 0,
	end_token_id: int = 2,
) -> torch.Tensor:
	"""Greedy decoding sampler."""
	batch_size = memory.size(1)
	generated = torch.full(
		(batch_size, 1), start_token_id, dtype=torch.long,
		device=memory.device
	)

	for _ in range(max_length - 1):
		dec = model.decode(generated, memory, tgt_mask=None, memory_mask=src_mask)
		last_dec = dec[-1]	# (batch, embed_dim)

		next_logits = model.output_projection(last_dec)	# (batch, vocab_size)
		next_tokens = next_logits.argmax(dim=-1).unsqueeze(1)

		generated = torch.cat([generated, next_tokens], dim=1)

		if (next_tokens == end_token_id).all():
			break

	return generated

def beam_search_sampler(
	model, memory: torch.Tensor, src_mask: torch.Tensor,
	max_length: int = 50, start_token_id: int = 0,
	end_token_id: int = 2, beam_size: int = 3,
) -> torch.Tensor:
	"""Beam search decoding sampler (batch size 1)."""
	generated = torch.tensor([[start_token_id]], device=memory.device)
	beams = [(generated, 0.0)]

	for _ in range(max_length - 1):
		new_beams = []
		for seq, score in beams:
			if seq[0, -1].item() == end_token_id:
				new_beams.append((seq, score))
				continue

			dec = model.decode(seq, memory, tgt_mask=None, memory_mask=src_mask)
			last_dec = dec[-1]	# (1, embed_dim)

			next_logits = model.output_projection(last_dec)	# (1, vocab_size)
			log_probs = F.log_softmax(next_logits, dim=-1)
			topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

			for k in range(beam_size):
				next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
				new_seq = torch.cat([seq, next_token], dim=1)
				new_score = score + topk_log_probs[0, k].item()
				new_beams.append((new_seq, new_score))

		beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
		if all(seq[0, -1].item() == end_token_id for seq, _ in beams):
			break

	best_seq, _ = max(beams, key=lambda x: x[1])
	return best_seq
