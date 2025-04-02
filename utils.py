import itertools
import os
import random
import numpy as np
import torch

def chunks(iterable, chunk_size):
	it = iter(iterable)
	while True:
		chunk = list(itertools.islice(it, chunk_size))
		if not chunk:
			break
		yield chunk

def set_seed(seed: int = 24) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)

def filter_none_kwargs(**kwargs):
	return {k: v for k, v in kwargs.items() if v is not None}
