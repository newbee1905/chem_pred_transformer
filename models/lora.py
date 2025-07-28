import torch																																					 
import torch.nn as nn																																			
import math																																					  

class LoRAInjectedLinear(nn.Module):
	"""
	A wrapper class that injects a LoRA layer into an existing nn.Linear layer.
	"""
	def __init__(self, original_layer: nn.Linear, rank: int, alpha: int = 8):
		super().__init__()
		
		self.original_layer = original_layer
		self.original_layer.requires_grad_(False)
		
		in_features, out_features = original_layer.in_features, original_layer.out_features
		self.in_features = in_features
		self.out_features = out_features
		self.rank = rank
		self.alpha = alpha

		self.lora_A = nn.Linear(in_features, rank, bias=False)
		self.lora_B = nn.Linear(rank, out_features, bias=False)
		
		# Init LoRA weights
		# A is initialized with Kaiming uniform, B is initialized to zero.
		# This ensures that at the start of training, the LoRA modification is zero.
		nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
		nn.init.zeros_(self.lora_B.weight)
		
		self.scaling = self.alpha / self.rank

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		original_output = self.original_layer(x)
		lora_output = self.lora_B(self.lora_A(x)) * self.scaling
		
		return original_output + lora_output

def apply_lora_to_model(model: nn.Module, rank: int, alpha: int):
	for name, module in model.named_children():
		if isinstance(module, nn.Linear):
			setattr(model, name, LoRAInjectedLinear(module, rank, alpha))
		else:
			apply_lora_to_model(module, rank, alpha)

