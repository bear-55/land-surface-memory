"""Data-free checks for the frozen protocol and hydrological preprocessing."""
import csv
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import DevelopmentStreamflow, valid_window_ends
from evaluate import basin_metrics
from evaluation import parse_streamflow
from preprocessing import apply_static, fit_dynamic, fit_static
from project import ROOT, SEEDS, config, digest, read_json, split_ids


class ProtocolTests(unittest.TestCase):
    def test_frozen_splits_and_feature_contract(self):
        splits = split_ids()
        manifest = read_json(ROOT / "splits/formal_pub_527/manifest.json")
        self.assertEqual({k: len(v) for k, v in splits.items()}, manifest["counts"])
        self.assertEqual(len(set(sum(splits.values(), []))), 527)
        for name, expected in manifest["sha256"].items():
            self.assertEqual(digest(ROOT / "splits/formal_pub_527" / name), expected)
        cfg = config()
        self.assertEqual(cfg["training"]["paired_seeds"], list(SEEDS))
        self.assertEqual(cfg["training"]["confirmation_seeds"], list(SEEDS))
        with (ROOT / cfg["features"]["manifest"]).open(encoding="utf-8-sig", newline="") as stream:
            features = list(csv.DictReader(stream))
        inputs = [r for r in features if r["allow_in_backbone"] == "true"]
        self.assertEqual(sum(r["group"] == "D" for r in inputs), 5)
        self.assertEqual(sum(r["group"] in ("C", "L") for r in inputs), 20)
        self.assertEqual(sum(r["group"] == "L" for r in inputs), 16)
        self.assertEqual(cfg["features"]["static_missing_indicators"], ["geol_porosity"])
        self.assertTrue(all(r["group"] == "L" for r in features if r["allow_in_controller"] == "true"))
        self.assertFalse(cfg["features"]["include_basin_id"])
        self.assertFalse(cfg["features"]["include_observed_q_history"])
        self.assertFalse(cfg["features"]["include_q_derived_attributes"])

    def test_static_fit_uses_only_training_basins(self):
        names = ["area", "geol_porosity"]
        raw = np.array([[1., .1], [3., np.nan], [7., .3], [100., .9]])
        fitted = fit_static(raw, names, [0, 1, 2], ["geol_porosity"], ["area"])
        changed = raw.copy()
        changed[3] = [1e8, 1e6]
        self.assertEqual(fitted, fit_static(changed, names, [0, 1, 2], ["geol_porosity"], ["area"]))
        self.assertAlmostEqual(fitted["median"][1], .2)
        transformed = apply_static(raw, fitted)
        self.assertEqual(transformed.shape, (4, 3))
        np.testing.assert_array_equal(transformed[:, -1], [0., 1., 0., 0.])
        self.assertAlmostEqual(float(transformed[1, 1]), 0., places=6)

    def test_dynamic_fit_excludes_other_basins_and_periods(self):
        dates = np.arange("2000-01-01", "2000-01-06", dtype="datetime64[D]")
        values = np.arange(30, dtype=float).reshape(3, 5, 2)
        valid = np.ones((3, 5), dtype=bool)
        fitted = fit_dynamic(values, valid, dates, [0, 1], "2000-01-01", "2000-01-03")
        changed = values.copy()
        changed[2] += 1e9
        changed[:2, 3:] -= 1e9
        self.assertEqual(fitted, fit_dynamic(changed, valid, dates, [0, 1], "2000-01-01", "2000-01-03"))
        self.assertEqual(fitted["n_training_days"], 6)
        np.testing.assert_allclose(fitted["mean"], values[:2, :3].reshape(-1, 2).mean(axis=0))

    def test_streamflow_units_flags_and_period(self):
        text = "\n".join([
            "00000001 1999 12 31 -999 M",
            "00000001 2000 1 1 1 A",
            "00000001 2000 1 2 2 A:e",
            "00000001 2000 1 3 999 M",
            "00000001 2000 1 4 -999 A",
            "00000001 2000 1 5 100 A"])
        dates, q = parse_streamflow(text, "00000001", ["2000-01-01", "2000-01-04"], 1e6)
        np.testing.assert_array_equal(dates, np.arange("2000-01-01", "2000-01-05", dtype="datetime64[D]"))
        np.testing.assert_allclose(q[:2], [2.4465755455488, 4.8931510910976], rtol=1e-7)
        self.assertTrue(np.isnan(q[2:]).all())
        with self.assertRaises(ValueError):
            parse_streamflow(text, "00000001", ["2000-01-01", "2000-01-04"], 0)

    def test_held_out_labels_rejected_before_file_lookup(self):
        splits = {"train": ["00000001"], "val": ["00000002"], "test": ["00000003"]}
        reader = DevelopmentStreamflow({}, splits, {r: ["2000-01-01", "2000-01-02"] for r in ("train", "val")})
        for role in ("train", "val", "test"):
            with self.assertRaises(PermissionError):
                reader.read("00000003", role, 1e6)
        with self.assertRaises(PermissionError):
            reader.read("00000001", "val", 1e6)
        self.assertEqual(reader.accesses, [])

    def test_metrics_keep_invalid_nse_and_reject_invalid_predictions(self):
        row = basin_metrics(np.zeros(365), np.ones(365))
        self.assertIsNone(row["nse"])
        self.assertEqual(row["nse_invalid_reason"], "near_zero_variance")
        self.assertEqual(row["mae_mm_day"], 1.)
        self.assertIsNone(row["volume_bias_percent"])
        self.assertEqual(basin_metrics([1, 2], [1, 2])["nse_invalid_reason"], "too_few_days")
        tied = basin_metrics([1, 2, 3, 3], [1, 2, 3, 3], min_days=4)
        self.assertEqual(tied["high_flow_days"], 2)
        self.assertEqual(tied["nse"], 1.)
        with self.assertRaises(FloatingPointError):
            basin_metrics([1, 2, 3], [1, np.nan, 3], min_days=3)

    def test_causal_windows_do_not_bridge_invalid_days(self):
        dates = np.arange("2000-01-01", "2000-01-09", dtype="datetime64[D]")
        valid = np.ones(8, dtype=bool)
        valid[3] = False
        ends = valid_window_ends(valid, dates, "2000-01-01", "2000-01-08", 3)
        np.testing.assert_array_equal(ends, [2, 6, 7])
        with self.assertRaises(ValueError):
            valid_window_ends(valid[:7], np.delete(dates, 3), "2000-01-01", "2000-01-08", 3)


if __name__ == "__main__":
    unittest.main()
