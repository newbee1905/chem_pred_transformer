import torch
import torchmetrics

class SMILESEvaluationMetric(torchmetrics.Metric):
	def __init__(self, dist_sync_on_step=False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("valid_count", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")
		self.add_state("tanimoto_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")

	def update(self, preds: list, targets: list) -> None:
		assert len(preds) == len(targets), "Predictions and targets must have the same length"

		valid_count = 0
		tanimoto_sum = 0.0
		tanimoto_count = 0

		for pred, target in zip(preds, targets):
			mol_pred = Chem.MolFromSmiles(pred)
			mol_target = Chem.MolFromSmiles(target)

			if mol_pred and mol_target:
				fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2, nBits=1024)
				fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2, nBits=1024)
				tanimoto_sum += DataStructs.TanimotoSimilarity(fp_pred, fp_target)
				valid_count += 1

		self.valid_count += valid_count
		self.total_count += len(preds)
		self.tanimoto_sum += tanimoto_sum

	def compute(self):
		valid_smiles_ratio = self.valid_count / self.total_count if self.total_count > 0 else torch.tensor(0.0)
		avg_tanimoto = self.tanimoto_sum / self.valid_count if self.valid_count > 0 else torch.tensor(0.0)

		return {"valid_smiles_ratio": valid_smiles_ratio, "avg_tanimoto": avg_tanimoto}

	def reset(self):
		self.valid_count.zero_()
		self.total_count.zero_()
		self.tanimoto_sum.zero_()
