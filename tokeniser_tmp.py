import torch
from tokenisers.chemformer import ChemformerTokenizer

if __name__ == "__main__":
	tokeniser = ChemformerTokenizer(filename="bart_vocab.json")
	smi = ["CCN1CCN(c2ccc(-c3nc(CCN)no3)cc2F)CC1", "C"]
	enc = tokeniser.encode(smi)
	print(enc)

	for i, e in enumerate(enc):
		current_len = e.size(0)
		if current_len < 64:
			pad_length = 64 - current_len
			pad_token_id = tokeniser.vocabulary[tokeniser.special_tokens["pad"]]
			pad_tensor = torch.full((pad_length,), pad_token_id)
			e = torch.cat([e, pad_tensor])
		enc[i] = e
	
	print(tokeniser.vocabulary[tokeniser.special_tokens["start"]])
	print(enc)

	dec = tokeniser.decode(enc)
	print(dec)

	print(tokeniser.special_tokens)
