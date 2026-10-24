from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from common import REPRO, RESULTS


class Phase6Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.statistics = json.loads((RESULTS / "statistics.json").read_text())
        cls.runtime = json.loads((RESULTS / "runtime.json").read_text())

    def test_status_and_declared_comparisons(self):
        self.assertEqual(self.statistics["status"], "complete")
        self.assertEqual(self.runtime["status"], "complete")
        self.assertEqual(len(self.statistics["manuscript_seed_mean_comparisons"]), 6)
        self.assertEqual(len(self.statistics["historical_seed0_comparisons"]), 4)
        self.assertEqual(len(self.runtime["runs"]), 30)

    def test_historical_seed0_dm_reconstruction(self):
        historical = json.loads((REPRO / "results/metrics.json").read_text())["aggregate"]["dm"]
        for row in self.statistics["historical_seed0_comparisons"]:
            expected = historical[row["baseline"]]
            observed = next(x for x in row["hac"] if x["lag"] == 48)
            self.assertAlmostEqual(observed["stat"], expected["stat"], places=4)
            self.assertTrue(math.isclose(observed["p_two_sided"], expected["p"], rel_tol=2e-5, abs_tol=1e-15))

    def test_manuscript_effects_reconcile(self):
        rows = {x["label"]: x for x in self.statistics["manuscript_seed_mean_comparisons"]}
        self.assertAlmostEqual(rows["aggregate_8_seed_mean_pv_vs_lightgbm-q"]["relative_pv_reduction_percent"], 6.821, places=3)
        self.assertAlmostEqual(rows["aggregate_48_seed_mean_pv_vs_lightgbm-q"]["relative_pv_reduction_percent"], -0.503, places=3)
        self.assertAlmostEqual(rows["household_8_pv_s0_vs_lightgbm-q"]["relative_pv_reduction_percent"], -0.087, places=3)

    def test_inference_sensitivity_is_complete_and_finite(self):
        for row in self.statistics["manuscript_seed_mean_comparisons"] + self.statistics["historical_seed0_comparisons"]:
            self.assertEqual([x["lag"] for x in row["hac"]], [0, 8, 24, 48, 96, 336])
            self.assertEqual([x["block"] for x in row["block_bootstrap"]], [48, 96, 336])
            for item in row["hac"]:
                self.assertTrue(np.isfinite(item["stat"]))
                self.assertGreaterEqual(item["p_two_sided"], 0)
                self.assertLessEqual(item["p_two_sided"], 1)
            for item in row["block_bootstrap"]:
                self.assertLessEqual(item["mean_ci95"][0], item["mean_ci95"][1])

    def test_runtime_samples_and_summaries(self):
        for row in self.runtime["runs"]:
            samples = np.asarray(row["samples_ms"])
            self.assertEqual(samples.size, 100)
            self.assertTrue(np.isfinite(samples).all())
            self.assertTrue((samples >= 0).all())
            self.assertAlmostEqual(row["median_ms"], float(np.median(samples)), places=12)
            self.assertAlmostEqual(row["p95_ms"], float(np.percentile(samples, 95)), places=12)

    def test_runtime_scope_rows_present(self):
        keys = {(x["component"], x["model"], x["series"], x["horizons"]) for x in self.runtime["runs"]}
        for series in (1, 300):
            self.assertIn(("features_plus_inference", "PV-TCQN", series, 48), keys)
            for horizons in (8, 48):
                self.assertIn(("buffer_level_update", "RW-ACI", series, horizons), keys)
                self.assertIn(("measured_compute_cycle", "PV-TCQN+RW-ACI", series, horizons), keys)


if __name__ == "__main__":
    unittest.main(verbosity=2)
