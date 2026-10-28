import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis  # noqa: E402


class PsiTests(unittest.TestCase):
    def test_identical_samples_have_zero_psi(self):
        rng = random.Random(1)
        data = [rng.gauss(0, 1) for _ in range(300)]
        self.assertAlmostEqual(analysis.population_stability_index(data, data), 0.0, places=9)

    def test_shifted_sample_is_significant(self):
        rng = random.Random(2)
        ref = [rng.gauss(0, 1) for _ in range(500)]
        cur = [rng.gauss(2, 1) for _ in range(500)]
        self.assertGreaterEqual(analysis.population_stability_index(ref, cur), analysis.PSI_SIGNIFICANT)

    def test_classify_thresholds(self):
        self.assertEqual(analysis.classify(0.05), "low change")
        self.assertEqual(analysis.classify(0.10), "moderate change")
        self.assertEqual(analysis.classify(0.25), "significant drift")


class PipelineTests(unittest.TestCase):
    def test_run_is_reproducible_and_writes_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = analysis.run(out_dir=a)
            second = analysis.run(out_dir=b)
            self.assertEqual(first, second)
            for name in ("data/reference.csv", "data/current.csv", "results/drift-report.json"):
                self.assertTrue((Path(a) / name).exists())
            self.assertEqual((Path(a) / "data/reference.csv").read_text(),
                             (Path(b) / "data/reference.csv").read_text())
            json.loads((Path(a) / "results/drift-report.json").read_text())

    def test_expected_drift_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            report = analysis.run(out_dir=d)
        features = report["features"]
        self.assertTrue(report["overall_drift_detected"])
        self.assertEqual(features["image_size_kb"]["status"], "low change")  # control feature
        for name in ("image_brightness", "image_sharpness", "top1_confidence", "inference_latency_ms"):
            self.assertGreaterEqual(features[name]["psi"], analysis.PSI_SIGNIFICANT)
        rates = report["rates"]
        self.assertGreater(rates["expert_referral"]["current_rate"], rates["expert_referral"]["reference_rate"])
        self.assertLess(rates["correct_prediction"]["current_rate"], rates["correct_prediction"]["reference_rate"])


if __name__ == "__main__":
    unittest.main()
