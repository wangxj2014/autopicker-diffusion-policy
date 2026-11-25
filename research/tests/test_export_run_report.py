from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_run_report import export_run


class ExportRunReportTests(unittest.TestCase):
    def test_exports_compact_logs_but_not_checkpoints_or_raw_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            (run / "episode_0000").mkdir(parents=True)
            (run / "summary.json").write_text('{"successes": 1}', encoding="utf-8")
            (run / "run_manifest.json").write_text('{"gpu": {}}', encoding="utf-8")
            (run / "system_metrics.csv").write_text("gpu_util_percent\n25\n", encoding="utf-8")
            (run / "checkpoint.pt").write_bytes(b"weights")
            (run / "episode_0000" / "episode.json").write_text('{"outcome": {"success": true}}', encoding="utf-8")
            (run / "episode_0000" / "states.npz").write_bytes(b"raw arrays")
            destination = root / "export"
            bundle = export_run(run, destination)
            self.assertTrue((destination / "summary.json").is_file())
            self.assertTrue((destination / "episode_0000" / "episode.json").is_file())
            self.assertFalse((destination / "checkpoint.pt").exists())
            self.assertFalse((destination / "episode_0000" / "states.npz").exists())
            self.assertEqual(bundle["schema"], "autopicker-diagnostic-bundle-v1")
            self.assertTrue((destination / "bundle_manifest.json").is_file())

    def test_refuses_to_overwrite_existing_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            (run / "summary.json").write_text("{}", encoding="utf-8")
            destination = root / "export"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                export_run(run, destination)


if __name__ == "__main__":
    unittest.main()
