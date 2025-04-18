import json
from tokenizers import Tokenizer, Regex, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from typing import List, Union
import torch

class ChemformerTokenizerFast(PreTrainedTokenizerFast):
	def __init__(self, config_path: str):
		with open(config_path, 'r', encoding='utf-8') as f:
			cfg = json.load(f)

		vocab = {tok: idx for idx, tok in enumerate(cfg['vocabulary'])}
		tokenizer = Tokenizer(
			models.WordLevel(
				vocab=vocab,
				unk_token=cfg['properties']['special_tokens']['unknown']
			)
		)

		tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(cfg['properties']['regex']), behavior="isolated")

		super().__init__(
			tokenizer_object=tokenizer,
			unk_token=   cfg['properties']['special_tokens']['unknown'],
			pad_token=   cfg['properties']['special_tokens']['pad'],
			bos_token=   cfg['properties']['special_tokens']['start'],
			eos_token=   cfg['properties']['special_tokens']['end'],
			mask_token=  cfg['properties']['special_tokens']['mask'],
			sep_token=   cfg['properties']['special_tokens']['sep'],
			additional_special_tokens=[]
		)

		self.chem_start_idx = cfg['properties']['chem_start_idx']

	def convert_tokens_to_string(self, tokens: List[str]) -> str:
		return "".join(tokens)

	def decode(
		self,
		token_ids: Union[List[int], torch.Tensor],
		skip_special_tokens: bool = False,
		**kwargs
	) -> str:
		toks = super().convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

		return self.convert_tokens_to_string(toks)

	def batch_decode(
		self,
		token_ids_list: Union[List[List[int]], torch.Tensor],
		skip_special_tokens: bool = False,
		**kwargs
	) -> List[str]:
		if isinstance(token_ids_list, torch.Tensor):
			token_ids_list = token_ids_list.tolist()

		decoded: List[str] = []
		for ids in token_ids_list:
			toks = self.convert_ids_to_tokens(
				ids,
				skip_special_tokens=skip_special_tokens
			)
			decoded.append(self.convert_tokens_to_string(toks))

		return decoded
