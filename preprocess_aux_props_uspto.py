import math
import pickle
import multiprocessing as mp

from tqdm import tqdm
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski

# property functions
prop_funcs = {
	"MolWt":							 Descriptors.MolWt,
	"LogP":								Crippen.MolLogP,
	"TPSA":								rdMolDescriptors.CalcTPSA,
	"NumHDonors":					Lipinski.NumHDonors,
	"NumHAcceptors":			 Lipinski.NumHAcceptors,
	"NumRotatableBonds":	 Lipinski.NumRotatableBonds,
	"RingCount":					 rdMolDescriptors.CalcNumRings,
}

def process_mol(mol):
	"""Return a dict of calculated props for one RDKit Mol."""
	if mol is None:
		return None
	out = {}
	for name, fn in prop_funcs.items():
		try:
			val = fn(mol)
			if not math.isfinite(val):
				continue
			out[name] = val
		except Exception:
			continue
	return out

def main():
	# load your list of Mol objects
	with open("data/uspto_mixed.pickle", "rb") as f:
		df = pickle.load(f)

	mols = []
	if "reactants_mol" in df.columns:
		mols.extend(df["reactants_mol"].tolist())
	if "products_mol" in df.columns:
		mols.extend(df["products_mol"].tolist())

	# init stats
	stats = {
		name: {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "count": 0}
		for name in prop_funcs
	}

	# parallelise
	pool = mp.Pool(4)
	try:
		for props in tqdm(pool.imap_unordered(process_mol, mols),
											total=len(mols),
											desc="Calculating properties"):
			if not props:
				continue
			for name, val in props.items():
				s = stats[name]
				s["min"] = min(s["min"], val)
				s["max"] = max(s["max"], val)
				s["sum"] += val
				s["count"]+= 1
	finally:
		pool.close()
		pool.join()

	# finalise means, drop sums
	for name, s in stats.items():
		if s["count"] > 0:
			s["mean"] = s["sum"] / s["count"]
		del s["sum"]

	# save as pickle
	with open("aux_prop_stats.pickle", "wb") as f:
		pickle.dump(stats, f)

	print("Saved property statistics to aux_prop_stats.pickle")

if __name__ == "__main__":
	main()
