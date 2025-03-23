import torch
from pprint import pp

from models.bart import BART

import torch
import torch.nn as nn
from tokenisers.chemformer import ChemformerTokenizer

def copy_embedding(src_weight, tgt_embedding, name="embedding"):
	if src_weight.shape[0] == tgt_embedding.weight.shape[0]:
		tgt_embedding.weight.data.copy_(src_weight)
		print(f"Copied {name} weights.")
	else:
		print(f"Skipping {name} weights: shape mismatch {src_weight.shape} vs {tgt_embedding.weight.shape}.")

def copy_linear(src_weight, src_bias, tgt_linear, name="linear"):
	if src_weight.shape == tgt_linear.weight.shape:
		tgt_linear.weight.data.copy_(src_weight)
		tgt_linear.bias.data.copy_(src_bias)
		print(f"Copied {name} weights directly.")
	else:
		print(f"Skipping {name}: shape mismatch {src_weight.shape} vs {tgt_linear.weight.shape}.")

def copy_norm(src_weight, src_bias, tgt_norm, name="norm"):
	if src_weight.shape == tgt_norm.weight.shape:
		tgt_norm.weight.data.copy_(src_weight)
		tgt_norm.bias.data.copy_(src_bias)
		print(f"Copied {name} weights directly.")
	else:
		print(f"Skipping {name}: shape mismatch {src_weight.shape} vs {tgt_norm.weight.shape}.")

def copy_feedforward_layer(src_state_dict, tgt_ff, layer_prefix):
	src_fc_in_w = src_state_dict[layer_prefix + "linear1.weight"]
	src_fc_in_b = src_state_dict[layer_prefix + "linear1.bias"]
	if src_fc_in_w.shape == tgt_ff.fc_in.weight.shape:
		tgt_ff.fc_in.weight.data.copy_(src_fc_in_w)
		tgt_ff.fc_in.bias.data.copy_(src_fc_in_b)
		print(f"Copied {layer_prefix} feedforward fc_in weights directly.")
	else:
		if (tgt_ff.fc_in.weight.shape[0] == src_fc_in_w.shape[0] * 2 and
				tgt_ff.fc_in.weight.shape[1] == src_fc_in_w.shape[1]):
			new_weight = torch.cat([src_fc_in_w, src_fc_in_w], dim=0)
			new_bias = torch.cat([src_fc_in_b, src_fc_in_b], dim=0)
			tgt_ff.fc_in.weight.data.copy_(new_weight)
			tgt_ff.fc_in.bias.data.copy_(new_bias)
			print(f"Adapted {layer_prefix} feedforward fc_in weights by duplication.")
		else:
			print(f"Skipping {layer_prefix} feedforward fc_in: shape mismatch {src_fc_in_w.shape} vs {tgt_ff.fc_in.weight.shape}.")
	src_fc_out_w = src_state_dict[layer_prefix + "linear2.weight"]
	src_fc_out_b = src_state_dict[layer_prefix + "linear2.bias"]
	if src_fc_out_w.shape == tgt_ff.fc_out.weight.shape:
		tgt_ff.fc_out.weight.data.copy_(src_fc_out_w)
		tgt_ff.fc_out.bias.data.copy_(src_fc_out_b)
		print(f"Copied {layer_prefix} feedforward fc_out weights directly.")
	else:
		print(f"Skipping {layer_prefix} feedforward fc_out: shape mismatch {src_fc_out_w.shape} vs {tgt_ff.fc_out.weight.shape}.")

def copy_mha_weights(src_prefix, tgt_mha, src_state_dict):
	d_model = tgt_mha.d_model
	in_proj_weight = src_state_dict[src_prefix + "in_proj_weight"]
	in_proj_bias = src_state_dict[src_prefix + "in_proj_bias"]
	tgt_mha.q_proj.weight.data.copy_(in_proj_weight[:d_model])
	tgt_mha.q_proj.bias.data.copy_(in_proj_bias[:d_model])
	tgt_mha.k_proj.weight.data.copy_(in_proj_weight[d_model:2*d_model])
	tgt_mha.k_proj.bias.data.copy_(in_proj_bias[d_model:2*d_model])
	tgt_mha.v_proj.weight.data.copy_(in_proj_weight[2*d_model:3*d_model])
	tgt_mha.v_proj.bias.data.copy_(in_proj_bias[2*d_model:3*d_model])
	tgt_mha.out_proj.weight.data.copy_(src_state_dict[src_prefix + "out_proj.weight"])
	tgt_mha.out_proj.bias.data.copy_(src_state_dict[src_prefix + "out_proj.bias"])
	print(f"Copied MHA weights from {src_prefix}.")

def copy_norm_layer(src_state_dict, src_prefix, tgt_norm, norm_name):
	src_w = src_state_dict[src_prefix + "weight"]
	src_b = src_state_dict[src_prefix + "bias"]
	copy_norm(src_w, src_b, tgt_norm, norm_name)

def copy_encoder_layer(i, ckpt_state_dict, model):
	layer_prefix = f"encoder.layers.{i}."
	enc_layer = model.enc_layers[i]
	copy_mha_weights(layer_prefix + "self_attn.", enc_layer.self_attn, ckpt_state_dict)
	copy_norm_layer(ckpt_state_dict, layer_prefix + "norm1.", enc_layer.self_attn_norm, f"encoder.layers.{i}.self_attn_norm")
	copy_norm_layer(ckpt_state_dict, layer_prefix + "norm2.", enc_layer.ff_norm, f"encoder.layers.{i}.ff_norm")
	copy_feedforward_layer(ckpt_state_dict, enc_layer.ff, layer_prefix)

def copy_decoder_layer(i, ckpt_state_dict, model):
	layer_prefix = f"decoder.layers.{i}."
	dec_layer = model.dec_layers[i]
	copy_mha_weights(layer_prefix + "self_attn.", dec_layer.self_attn, ckpt_state_dict)
	copy_norm_layer(ckpt_state_dict, layer_prefix + "norm1.", dec_layer.self_attn_norm, f"decoder.layers.{i}.self_attn_norm")
	copy_mha_weights(layer_prefix + "multihead_attn.", dec_layer.cross_attn, ckpt_state_dict)
	copy_norm_layer(ckpt_state_dict, layer_prefix + "norm2.", dec_layer.cross_attn_norm, f"decoder.layers.{i}.cross_attn_norm")
	copy_norm_layer(ckpt_state_dict, layer_prefix + "norm3.", dec_layer.ff_norm, f"decoder.layers.{i}.ff_norm")
	copy_feedforward_layer(ckpt_state_dict, dec_layer.ff, layer_prefix)

def copy_final_layer(ckpt_state_dict, model):
	if "token_fc.weight" in ckpt_state_dict and "token_fc.bias" in ckpt_state_dict:
		orig_token_fc_w = ckpt_state_dict["token_fc.weight"]
		orig_token_fc_b = ckpt_state_dict["token_fc.bias"]
		if orig_token_fc_w.shape == tuple(model.fc_out.weight.shape):
			model.fc_out.weight.data.copy_(orig_token_fc_w)
			model.fc_out.bias.data.copy_(orig_token_fc_b)
			print("Copied final token_fc weights to fc_out.")
		else:
			print(f"Skipping final token_fc: shape mismatch {orig_token_fc_w.shape} vs {model.fc_out.weight.shape}.")
	else:
		print("No final token_fc found in checkpoint.")

def copy_all_weights(ckpt, model):
	ckpt_state_dict = ckpt["state_dict"]
	if "emb.weight" in ckpt_state_dict:
		copy_embedding(ckpt_state_dict["emb.weight"], model.emb, "embedding")
	else:
		print("Embedding 'emb.weight' not found in checkpoint.")
	num_enc_layers = len(model.enc_layers)
	for i in range(num_enc_layers):
		print(f"Copying encoder layer {i}...")
		copy_encoder_layer(i, ckpt_state_dict, model)
	num_dec_layers = len(model.dec_layers)
	for i in range(num_dec_layers):
		print(f"Copying decoder layer {i}...")
		copy_decoder_layer(i, ckpt_state_dict, model)
	copy_final_layer(ckpt_state_dict, model)
	print("Finished copying all model weights.")

if __name__ == "__main__":
	chemformer_small_state_dict = torch.load("step=1000000.ckpt", weights_only=False)

	tokeniser = ChemformerTokenizer(filename="bart_vocab.json")

	model = BART(
		vocab_size=len(tokeniser),
		norm_layer=nn.LayerNorm,
		d_model=512,
		n_heads=8,
		n_layers=6,
		d_ff=2048,
		activation="gelu",
	)
	print(model)

	copy_all_weights(chemformer_small_state_dict, model)

	torch.save(model.state_dict(), "chemformer_small.pth")
