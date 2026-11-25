import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autopicker_dp.telemetry import ResourceMonitor, _process_tree_snapshot, write_run_manifest


class TelemetryTests(unittest.TestCase):
    @mock.patch('autopicker_dp.telemetry._gpu_static', return_value={'available': False, 'devices': []})
    def test_manifest_records_host_git_configuration_and_input_hash(self, _gpu_static):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.json"
            source.write_text('{"fixture": true}', encoding="utf-8")
            output = root / "run"
            manifest = write_run_manifest(
                output, root, "unit_test", configuration={"seed": 7},
                input_paths={"fixture": source}, extra={"note": "test"})
            saved = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(saved["schema"], "autopicker-run-manifest-v1")
            self.assertEqual(saved["run_type"], "unit_test")
            self.assertEqual(saved["configuration"]["seed"], 7)
            self.assertEqual(saved["inputs"]["fixture"]["sha256"], manifest["inputs"]["fixture"]["sha256"])
            self.assertIn("gpu", saved)
            self.assertIn("host", saved)

    def test_process_tree_snapshot_includes_current_process(self):
        snapshot = _process_tree_snapshot(__import__("os").getpid())
        self.assertGreaterEqual(snapshot["process_tree_count"], 1)
        self.assertGreaterEqual(snapshot["process_tree_rss_mb"], 0)
        self.assertGreaterEqual(snapshot["process_tree_cpu_s"], 0)

    @mock.patch("autopicker_dp.telemetry._gpu_dynamic", return_value=([], None))
    @mock.patch("autopicker_dp.telemetry._gpu_processes", return_value=([], None))
    def test_monitor_writes_csv_and_returns_aggregates(self, _gpu_processes, _gpu_dynamic):
        with tempfile.TemporaryDirectory() as folder:
            monitor = ResourceMonitor(folder, interval_s=0.05).start()
            time.sleep(0.18)
            summary = monitor.stop()
            csv_path = Path(folder) / "system_metrics.csv"
            self.assertTrue(csv_path.is_file())
            rows = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(rows), 2)
            self.assertIn("system_cpu_percent", rows[0])
            self.assertIn("process_tree_rss_mb", rows[0])
            self.assertGreaterEqual(summary["sample_count"], 1)
            self.assertEqual(summary["metrics_csv"], "system_metrics.csv")

    @mock.patch("autopicker_dp.telemetry._gpu_dynamic", return_value=([], None))
    @mock.patch("autopicker_dp.telemetry._gpu_processes", return_value=([], None))
    def test_stop_takes_final_sample_for_subsecond_run(self, _gpu_processes, _gpu_dynamic):
        with tempfile.TemporaryDirectory() as folder:
            monitor = ResourceMonitor(folder, interval_s=10.0).start()
            time.sleep(0.05)
            summary = monitor.stop()
            rows = (Path(folder) / "system_metrics.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)  # header + immediate sample + final sample
            self.assertGreaterEqual(summary["sample_count"], 2)
            self.assertEqual(summary["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
