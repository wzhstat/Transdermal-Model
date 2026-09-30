"""Lightweight AD checks; these do not load the neural-network checkpoints."""

import unittest
from pathlib import Path

import numpy as np

from ad import KpHumanAD


ROOT = Path(__file__).resolve().parents[1]


class KpHumanADTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ad = KpHumanAD(ROOT)
        cls.smiles = ["C[Si](C)(C)C"]
        cls.contributions = np.zeros((len(cls.ad.seeds), 1, 128), dtype=np.float64)

    def test_default_is_025_and_calibration_is_preserved(self):
        result = self.ad.score(self.smiles, self.contributions)
        self.assertEqual(result["ad_structure_threshold_tanimoto"][0], 0.25)
        self.assertAlmostEqual(self.ad.calibrated_structure_q05, 0.23350877192982455)
        self.assertEqual(result["ad_nearest_train_parent"].shape, (1,))

    def test_custom_structure_cutoff_changes_flag(self):
        loose = self.ad.score(self.smiles, self.contributions, min_tanimoto=0)
        strict = self.ad.score(self.smiles, self.contributions, min_tanimoto=1)
        self.assertTrue(loose["ad_structure_in_domain"][0])
        self.assertFalse(strict["ad_structure_in_domain"][0])
        self.assertEqual(strict["ad_structure_threshold_tanimoto"][0], 1)

    def test_custom_model_cutoffs_change_combined_flag(self):
        loose = self.ad.score(
            self.smiles, self.contributions,
            min_tanimoto=0, max_nn_ratio=1e9, max_q_ratio=1e9,
        )
        strict = self.ad.score(
            self.smiles, self.contributions,
            min_tanimoto=0, max_nn_ratio=0, max_q_ratio=0,
        )
        self.assertTrue(loose["ad_screen_in_domain"][0])
        self.assertFalse(strict["ad_screen_in_domain"][0])

    def test_invalid_cutoff_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ad.score(self.smiles, self.contributions, min_tanimoto=1.01)
        with self.assertRaises(ValueError):
            self.ad.score(self.smiles, self.contributions, max_nn_ratio=-1)

    def test_empty_input(self):
        contributions = np.empty((len(self.ad.seeds), 0, 128))
        result = self.ad.score([], contributions)
        self.assertEqual(len(result["ad_structure_in_domain"]), 0)


if __name__ == "__main__":
    unittest.main()
