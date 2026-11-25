"""Export a small, GitHub-friendly diagnostic bundle from an ignored run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "run_manifest.json", "summary.json", "report.json", "metrics.jsonl",
    "system_metrics.csv", "resource_monitor_errors.jsonl", "inference.log", "process.log",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_run(run_dir, destination):
    run_dir = Path(run_dir).resolve()
    destination = Path(destination).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {destination}")
    if destination == run_dir or run_dir in destination.parents:
        raise ValueError("Export destination must not be inside the source run directory")

    destination.mkdir(parents=True)
    copied = []
    skipped = []
    candidates = [run_dir / name for name in ROOT_FILES]
    for episode_dir in sorted(run_dir.glob("episode_*")):
        if episode_dir.is_dir():
            for name in ("episode.json", "failure.json"):
                candidate = episode_dir / name
                if candidate.is_file():
                    candidates.append(candidate)
    for source in candidates:
        if not source.is_file():
            continue
        relative = source.relative_to(run_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = source.stat().st_size
        if size > 20 * 1024 * 1024:
            skipped.append({"path": relative.as_posix(), "reason": "larger_than_20MiB", "size_bytes": size})
            continue
        shutil.copy2(source, target)
        copied.append({"path": relative.as_posix(), "size_bytes": size, "sha256": sha256(target)})

    if not copied:
        shutil.rmtree(destination)
        raise ValueError("No supported run logs found; expected run_manifest.json, summary.json or metrics.jsonl")
    readme = [
        "# Experiment diagnostic bundle",
        "",
        f"Source run: {run_dir.name}",
        "",
        "This bundle contains compact configuration, outcome, inference and resource logs.",
        "It intentionally excludes model checkpoints, raw datasets, RGB-D frames, videos and trajectory NPZ files.",
        "Files larger than 20 MiB are listed as skipped in bundle_manifest.json.",
        "",
    ]
    (destination / "BUNDLE_README.md").write_text("\n".join(readme), encoding="utf-8")
    bundle = {
        "schema": "autopicker-diagnostic-bundle-v1",
        "source_run_name": run_dir.name,
        "files": copied,
        "skipped": skipped,
    }
    (destination / "bundle_manifest.json").write_text(
        json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing runs/... directory")
    parser.add_argument("--destination", type=Path, required=True,
                        help="New tracked folder, e.g. docs/research/results/m7_policy_val_s0")
    options = parser.parse_args()
    bundle = export_run(options.run, options.destination)
    print("AUTOPICKER_DIAGNOSTIC_EXPORT_PASS " + json.dumps({
        "destination": str(options.destination), "files": len(bundle["files"]),
        "skipped": bundle["skipped"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
