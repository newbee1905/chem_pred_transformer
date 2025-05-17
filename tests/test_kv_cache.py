import torch
import pytest

from models.bart import BART
from models.sampler import greedy_sampler, beam_search_sampler

import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(autouse=True)
# Ensure reproducibility across all tests
def seed_everything():
	torch.manual_seed(0)

@pytest.fixture
def bart_model():
  model = BART(
    vocab_size=128,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    norm_layer=torch.nn.RMSNorm,
    activation="swiglu",
    theta=5000,
		max_seq_len=282,
  ).to(DEVICE)
  model.eval()

  return model

def test_kv_cache_equivalence_greedy_sampler(bart_model):
	"""Greedy-Sampler: Ensure KV-cache stepwise generation matches full-sequence generation."""
	bsz = 2
	src_len = 10
	tgt_len = 8

	src = torch.randint(0, bart_model.vocab_size, (bsz, src_len), device=DEVICE)
	src_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=DEVICE)

	out_full = bart_model.generate(
		src, src_mask, greedy_sampler,
		max_length=tgt_len * 2,
		start_token_id=0,
		end_token_id=1,
		kv_cache=False,
	)
	assert out_full.shape == (bsz, tgt_len * 2)

	out_cache = bart_model.generate(
		src, src_mask, greedy_sampler,
		max_length=tgt_len * 2,
		start_token_id=0,
		end_token_id=1,
		kv_cache=True,
	)
	assert out_cache.shape == (bsz, tgt_len * 2)

	assert torch.equal(out_full, out_cache), "Greedy Sampler: KV-cache output differs from full generation"

def test_kv_cache_equivalence_beam_search_sampler(bart_model):
	"""Beam-Search-Sampler: Ensure KV-cache stepwise generation matches full-sequence generation."""
	bsz = 2
	src_len = 10
	tgt_len = 8
	beam_size = 3

	src = torch.randint(0, bart_model.vocab_size, (bsz, src_len), device=DEVICE)
	src_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=DEVICE)

	seq_full, scores_full = bart_model.generate(
		src, src_mask, beam_search_sampler,
		max_length=tgt_len * 2,
		start_token_id=0,
		end_token_id=1,
		beam_size=beam_size,
		length_penalty_alpha=1.0,
		kv_cache=False,
	)
	assert seq_full.shape == (bsz, beam_size, tgt_len * 2)

	seq_cache, scores_cache = bart_model.generate(
		src, src_mask, beam_search_sampler,
		max_length=tgt_len * 2,
		start_token_id=0,
		end_token_id=1,
		beam_size=3,
		length_penalty_alpha=1.0,
		kv_cache=True,
	)
	assert seq_cache.shape == (bsz, beam_size, tgt_len * 2)

	assert torch.equal(seq_full, seq_cache),	"Greedy Sampler: KV-cache output differs from full generation"
	assert torch.allclose(scores_full, scores_cache, atol=1e-6), "Beam-search scores differ with KV-cache"


def test_kv_cache_generation_benchmark(bart_model, benchmark):
	"""Benchmark full-sequence greedy generation."""
	bsz = 64
	src_len = 282
	tgt_len = 282

	src = torch.randint(0, bart_model.vocab_size, (bsz, src_len), device=DEVICE)
	src_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=DEVICE)

	def kv_cache_gen():
		_ = bart_model.generate(
			src, src_mask, greedy_sampler,
			max_length=tgt_len,
			start_token_id=0,
			end_token_id=1,
			kv_cache=True,
		)

	benchmark(kv_cache_gen)

def test_full_generation_benchmark(bart_model, benchmark):
	"""Benchmark full-sequence greedy generation."""
	bsz = 64
	src_len = 282
	tgt_len = 282

	src = torch.randint(0, bart_model.vocab_size, (bsz, src_len), device=DEVICE)
	src_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=DEVICE)

	def full_gen():
		_ = bart_model.generate(
			src, src_mask, greedy_sampler,
			max_length=tgt_len,
			start_token_id=0,
			end_token_id=1,
			kv_cache=False,
		)

	benchmark(full_gen)
