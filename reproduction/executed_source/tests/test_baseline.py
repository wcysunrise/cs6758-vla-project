"""Offline protocol/integrity tests. These are not robot experiments."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "scripts/run_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = baseline.load_config(ROOT / "configs/baseline.yaml")

    def test_smoke_excluded_from_evaluation(self):
        evaluation = baseline.episode_plan(self.cfg)
        smoke = baseline.episode_plan(self.cfg, True)
        self.assertEqual(len(evaluation), 5)
        self.assertEqual(smoke, [(6, 1005, 5)])
        self.assertFalse(set(evaluation) & set(smoke))
        self.assertEqual(len({x[2] for x in evaluation + smoke}), 6)

    def test_invalid_protocol_is_rejected(self):
        for section, key, value in [("policy", "revision", "main"),
                                    ("policy", "dtype", "bfloat16"),
                                    ("environment", "task_id", 3),
                                    ("evaluation", "episodes", 100),
                                    ("evaluation", "initial_state_start", -1)]:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    baseline.load_config(path)

    def test_csv_rejects_duplicate_and_keeps_first_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.csv"
            row = dict.fromkeys(baseline.FIELDS, "")
            row.update(episode_id=1, status="completed", success=False)
            baseline.append_result(path, row)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                baseline.append_result(path, row)
            self.assertEqual(path.read_bytes(), original)
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_smoke_gate_rejects_modified_video_or_config(self):
        # Integrity fixture only; these bytes are never presented as a robot video.
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "fixture.bin"
            video.write_bytes(b"integrity test fixture")
            marker = Path(tmp) / "smoke.json"
            data = {"status": "completed", "video_file": video.name,
                    "video_sha256": baseline.sha256(video),
                    "config_sha256": baseline.config_fingerprint(self.cfg)}
            marker.write_text(json.dumps(data))
            baseline.verify_smoke(marker, self.cfg)
            changed_cfg = copy.deepcopy(self.cfg)
            changed_cfg["policy"]["n_action_steps"] = 10
            with self.assertRaises(ValueError):
                baseline.verify_smoke(marker, changed_cfg)
            video.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                baseline.verify_smoke(marker, self.cfg)


if __name__ == "__main__":
    unittest.main()
