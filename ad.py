"""Train-calibrated Kp/Human structural AD and optional exploratory screen.

Cutoffs describe training-set coverage, not prediction accuracy.
"""

import math
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class KpHumanAD:
    DEFAULT_MIN_TANIMOTO = 0.25

    def __init__(self, folder: Path):
        data = np.load(folder / "weights/kp_human_ad.npz", allow_pickle=False)
        if str(data["schema"]) != "skin-kp-human-ad-v1":
            raise ValueError("Unsupported AD reference schema")
        self.calibrated_structure_q05 = float(data["structure_threshold"])
        self.train_structure_smiles = data["train_structure_smiles"].astype(str).tolist()
        self.train_structure_parents = data["train_structure_parents"].astype(str).tolist()
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.train_fingerprints = [
            self.generator.GetFingerprint(Chem.MolFromSmiles(s))
            for s in self.train_structure_smiles
        ]
        self.seeds = []
        for i in range(int(data["n_seeds"])):
            self.seeds.append({
                "mean": data[f"seed_{i}_mean"].astype(np.float64),
                "components": data[f"seed_{i}_components"].astype(np.float64),
                "variance": data[f"seed_{i}_variance"].astype(np.float64),
                "train_pc_whitened": data[f"seed_{i}_train_pc_whitened"].astype(np.float64),
                "nn_q95": float(data[f"seed_{i}_nn_q95"]),
                "q_q95": float(data[f"seed_{i}_q_q95"]),
            })

    def score(
        self,
        smiles: list[str],
        contributions: np.ndarray,
        *,
        min_tanimoto: float | None = None,
        max_nn_ratio: float = 1.0,
        max_q_ratio: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Score against fixed training references with configurable decision cutoffs."""
        structure_threshold = (
            self.DEFAULT_MIN_TANIMOTO
            if min_tanimoto is None else float(min_tanimoto)
        )
        if not math.isfinite(structure_threshold) or not 0 <= structure_threshold <= 1:
            raise ValueError("min_tanimoto must be finite and between 0 and 1")
        for name, value in (("max_nn_ratio", max_nn_ratio), ("max_q_ratio", max_q_ratio)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if contributions.shape[:2] != (len(self.seeds), len(smiles)):
            raise ValueError("Contribution matrix must have shape [seed, molecule, feature]")
        if not smiles:
            return {
                "ad_structure_tanimoto": np.empty(0, dtype=float),
                "ad_structure_threshold_tanimoto": np.empty(0, dtype=float),
                "ad_nearest_train_parent": np.empty(0, dtype=str),
                "ad_structure_in_domain": np.empty(0, dtype=bool),
                "ad_model_nn_ratio": np.empty(0, dtype=float),
                "ad_model_q_ratio": np.empty(0, dtype=float),
                "ad_model_nn_threshold": np.empty(0, dtype=float),
                "ad_model_q_threshold": np.empty(0, dtype=float),
                "ad_model_in_domain": np.empty(0, dtype=bool),
                "ad_screen_in_domain": np.empty(0, dtype=bool),
            }
        similarity, nearest_parent = [], []
        for value in smiles:
            fingerprint = self.generator.GetFingerprint(Chem.MolFromSmiles(value))
            scores = DataStructs.BulkTanimotoSimilarity(fingerprint, self.train_fingerprints)
            nearest = int(np.argmax(scores))
            similarity.append(float(scores[nearest]))
            nearest_parent.append(self.train_structure_parents[nearest])
        similarity = np.asarray(similarity, dtype=np.float64)
        nn_ratios, q_ratios = [], []
        for seed, values in zip(self.seeds, contributions, strict=True):
            centered = values.astype(np.float64) - seed["mean"]
            scores = centered @ seed["components"].T
            whitened = scores / np.sqrt(seed["variance"])
            reference = seed["train_pc_whitened"]
            distances_sq = (
                np.sum(whitened * whitened, axis=1)[:, None]
                + np.sum(reference * reference, axis=1)[None, :]
                - 2.0 * whitened @ reference.T
            )
            nearest_distance = np.sqrt(
                np.maximum(distances_sq.min(axis=1), 0.0) / whitened.shape[1]
            )
            reconstruction = scores @ seed["components"]
            residual = np.sum((centered - reconstruction) ** 2, axis=1)
            nn_ratios.append(nearest_distance / seed["nn_q95"])
            q_ratios.append(residual / seed["q_q95"])
        nn_ratio = np.median(np.stack(nn_ratios), axis=0)
        q_ratio = np.median(np.stack(q_ratios), axis=0)
        structural_in = similarity >= structure_threshold
        functional_in = (nn_ratio <= max_nn_ratio) & (q_ratio <= max_q_ratio)
        return {
            "ad_structure_tanimoto": similarity,
            "ad_structure_threshold_tanimoto": np.full(
                len(smiles), structure_threshold, dtype=np.float64
            ),
            "ad_nearest_train_parent": np.asarray(nearest_parent, dtype=str),
            "ad_structure_in_domain": structural_in,
            "ad_model_nn_ratio": nn_ratio,
            "ad_model_q_ratio": q_ratio,
            "ad_model_nn_threshold": np.full(len(smiles), max_nn_ratio, dtype=np.float64),
            "ad_model_q_threshold": np.full(len(smiles), max_q_ratio, dtype=np.float64),
            "ad_model_in_domain": functional_in,
            "ad_screen_in_domain": structural_in & functional_in,
        }
