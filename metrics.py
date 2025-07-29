from torch._dynamo import disable
import torch
import torchmetrics

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class SMILESEvaluationMetric(torchmetrics.Metric):
	@disable
	def __init__(self, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("valid_count", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.unique_smiles_set = set()
		self.add_state("duplicates_count", default=torch.tensor(0), dist_reduce_fx="sum")
		self.mfpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

	@disable
	def update(self, preds: list, targets: list) -> None:
		assert len(preds) == len(targets), "Predictions and targets must have the same length"

		device = self.valid_count.device
		valid_count = 0
		tanimoto_sum = 0.0
		duplicates = 0

		for pred, target in zip(preds, targets):
			mol_pred = Chem.MolFromSmiles(pred)
			mol_target = Chem.MolFromSmiles(target)

			if mol_pred and mol_target:
				fp_pred = self.mfpgen.GetFingerprint(mol_pred)
				fp_target = self.mfpgen.GetFingerprint(mol_target)
				tanimoto_sum += DataStructs.TanimotoSimilarity(fp_pred, fp_target)

				if pred in self.unique_smiles_set:
					duplicates += 1
				else:
					self.unique_smiles_set.add(pred)

				valid_count += 1

		self.valid_count += valid_count
		self.total_count += len(preds)
		self.tanimoto_sum += tanimoto_sum
		self.duplicates_count += duplicates

	@disable
	def compute(self):
		unique_smiles_count = torch.tensor(len(self.unique_smiles_set), dtype=torch.float)

		valid_smiles_ratio = self.valid_count / self.total_count if self.total_count > 0 else torch.tensor(0.0)
		avg_tanimoto = self.tanimoto_sum / self.total_count if self.valid_count > 0 else torch.tensor(0.0)
		unique_ratio = unique_smiles_count / self.total_count if self.total_count > 0 else torch.tensor(0.0)
		duplicate_ratio = self.duplicates_count / self.total_count if self.total_count > 0 else torch.tensor(0.0)

		return {
			"valid_smiles_ratio": valid_smiles_ratio,
			"avg_tanimoto": avg_tanimoto,
			"unique_ratio": unique_ratio,
			"duplicate_ratio": duplicate_ratio,
			"duplicate_count": self.duplicates_count.to(torch.float16),
		}

	@disable
	def compute_once(self, preds: list, targets: list) -> float:
		tanimoto_sum = 0.0
		valid_count = 0
		exact_match_count = 0

		for pred, target in zip(preds, targets):
			if pred == target:
				exact_match_count += 1

			mol_pred = Chem.MolFromSmiles(pred)
			mol_target = Chem.MolFromSmiles(target)

			if mol_pred and mol_target:
				fp_pred = self.mfpgen.GetFingerprint(mol_pred)
				fp_target = self.mfpgen.GetFingerprint(mol_target)

				tanimoto_sum += DataStructs.TanimotoSimilarity(fp_pred, fp_target)
				valid_count += 1

		total_count = len(preds)
		
		# Ensure division by zero is handled, return 0 if no valid pairs
		avg_tanimoto_val = tanimoto_sum / total_count if total_count > 0 else 0.0
		valid_smiles_ratio_val = valid_count / total_count if total_count > 0 else 0.0
		exact_match_ratio_val = exact_match_count / total_count if total_count > 0 else 0.0

		return {
			"avg_tanimoto": avg_tanimoto_val,
			"valid_smiles_ratio": valid_smiles_ratio_val,
			"exact_match_ratio": exact_match_ratio_val,
			"total_count": total_count,
		}

	@disable
	def reset(self):
		self.valid_count.zero_()
		self.total_count.zero_()
		self.tanimoto_sum.zero_()
		self.unique_smiles_set.clear()
		self.duplicates_count.zero_()


_mfpgen_singleton = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

@disable
def compute_batch_tanimoto_rewards(
	pred_smiles_list: list[str], 
	target_smiles_list: list[str],
	device: torch.device = torch.device("cpu")
) -> torch.Tensor:
	"""
	Computes Tanimoto similarity for a batch of predicted and target SMILES strings.
	Returns a tensor of rewards.
	"""
	if len(pred_smiles_list) != len(target_smiles_list):
		raise ValueError("preds and targets must have the same length")

	rewards = []
	for pred_smi, target_smi in zip(pred_smiles_list, target_smiles_list):
		mol_pred = Chem.MolFromSmiles(pred_smi)
		mol_target = Chem.MolFromSmiles(target_smi)
		
		reward = 0.0
		if mol_pred and mol_target:
			try:
				fp_pred = _mfpgen_singleton.GetFingerprint(mol_pred)
				fp_target = _mfpgen_singleton.GetFingerprint(mol_target)
				reward = DataStructs.TanimotoSimilarity(fp_pred, fp_target)
			except Exception:
				pass
		rewards.append(reward)
	
	return torch.tensor(rewards, dtype=torch.float32, device=device)
