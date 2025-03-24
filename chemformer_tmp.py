import torch
from pprint import pp

from models.bart import BART
from models.chemformer import Chemformer

import torch
import torch.nn as nn
from tokenisers.chemformer import ChemformerTokenizer

if __name__ == "__main__":
	tokeniser = ChemformerTokenizer(filename="bart_vocab.json")
	chemformer_small_state_dict = torch.load("step=1000000.ckpt", weights_only=False)["state_dict"]

	model = Chemformer(
		vocab_size=len(tokeniser),
		norm_layer=nn.LayerNorm,
		d_model=512,
		n_heads=8,
		n_layers=6,
		d_ff=2048,
		activation="gelu",
	)
	model.load_state_dict(chemformer_small_state_dict)
	torch.save(model.state_dict(), "chemformer_small_2.pth")
