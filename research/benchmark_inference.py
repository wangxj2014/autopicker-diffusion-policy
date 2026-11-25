"""Benchmark DDIM inference-step latency and action deviation on a fixed checkpoint.

This measures policy forward-pass latency on the selected device. It does not
include the separate inference worker IPC, Isaac Sim stepping, camera latency,
or ROS 2 control latency measured by a full rollout.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from autopicker_dp.dataset import Normalizer, SequenceDataset, load_dataset
from autopicker_dp.telemetry import ResourceMonitor, write_run_manifest
from autopicker_dp.train import load_policy


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional latency array")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def choose_validation_windows(sequence_dataset, episode_ends, max_samples):
    """Select windows evenly within each selected episode, then cap the total."""
    grouped = defaultdict(list)
    for index, (start, end, current) in enumerate(sequence_dataset.indices):
        episode = int(np.searchsorted(episode_ends, end))
        grouped[episode].append(index)
    if not grouped:
        raise ValueError("No sequence windows available for benchmark")
    episode_ids = sorted(grouped)
    quota = max(1, int(np.ceil(max_samples / len(episode_ids))))
    chosen = []
    for episode in episode_ids:
        candidates = grouped[episode]
        count = min(quota, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, num=count, dtype=np.int64)
        chosen.extend(candidates[position] for position in positions)
    if len(chosen) > max_samples:
        positions = np.linspace(0, len(chosen) - 1, num=max_samples, dtype=np.int64)
        chosen = [chosen[position] for position in positions]
    return chosen


def benchmark(options):
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(options.device)
    dataset = load_dataset(options.dataset)
    indices = dataset.episode_indices(options.split)
    if not indices:
        raise ValueError(f"No episodes in split {options.split!r}")
    sequences = SequenceDataset(dataset.data, dataset.episode_ends, indices)
    sample_indices = choose_validation_windows(sequences, dataset.episode_ends, options.max_samples)
    windows = []
    for sequence_index in sample_indices:
        window = sequences[sequence_index]
        windows.append({
            "sequence_index": int(sequence_index),
            "episode_index": int(np.searchsorted(dataset.episode_ends, sequences.indices[sequence_index][1])),
            "observation": window["obs"],
        })

    checkpoint_sha = sha256(options.checkpoint)
    dataset_sha = sha256(options.dataset / "manifest.json")
    policy, metadata = load_policy(options.checkpoint, options.upstream, device)
    if metadata["dataset_sha256"] != dataset_sha:
        raise ValueError("Checkpoint dataset hash differs from the selected dataset")
    if metadata["dt"] != dataset.manifest["dt"]:
        raise ValueError("Checkpoint action rate differs from selected dataset")
    normalizer = Normalizer.from_dict(metadata["normalizer"])
    tensors = [
        torch.as_tensor(normalizer.normalize(item["observation"], key="obs"),
                        dtype=torch.float32, device=device).unsqueeze(0)
        for item in windows
    ]

    steps_list = sorted(set(int(value) for value in options.steps) | {int(options.reference_steps)})
    if any(value < 1 or value > 100 for value in steps_list):
        raise ValueError("DDIM inference steps must be within [1, 100]")
    options.output.mkdir(parents=True, exist_ok=False)
    manifest = write_run_manifest(
        options.output, ROOT, "inference_step_benchmark",
        configuration={
            "dataset": str(options.dataset), "checkpoint": str(options.checkpoint),
            "upstream": str(options.upstream), "split": options.split,
            "device": str(device), "steps": steps_list,
            "reference_steps": options.reference_steps,
            "warmup": options.warmup, "repeats": options.repeats,
            "max_samples": options.max_samples, "seed": options.seed,
        },
        input_paths={"dataset_manifest": options.dataset / "manifest.json",
                     "checkpoint": options.checkpoint},
        extra={"dataset_sha256": dataset_sha, "checkpoint_sha256": checkpoint_sha,
               "training_git_revision": metadata.get("git_revision"),
               "training_git_dirty": metadata.get("git_dirty"),
               "training_split": metadata.get("split"),
               "training_seed": metadata.get("seed"),
               "training_observation_augmentation": metadata.get("observation_augmentation")})
    manifest_sha = sha256(options.output / "run_manifest.json")

    resource_monitor = ResourceMonitor(options.output, interval_s=1.0).start()
    timings = {steps: [] for steps in steps_list}
    outputs = {steps: {} for steps in steps_list}
    cuda_peak_allocated = {}
    cuda_peak_reserved = {}
    records = []
    try:
        for steps in steps_list:
            policy.set_inference_steps(steps)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for tensor_index, sample in enumerate(windows):
                sample_index = sample["sequence_index"]
                observation = tensors[tensor_index]
                for warmup_index in range(options.warmup):
                    policy.predict(observation, seed=options.seed + sample_index + warmup_index * 100000)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                for repeat in range(options.repeats):
                    seed = options.seed + sample_index
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    prediction = policy.predict(observation, seed=seed)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    normalized_chunk = policy.action_chunk(prediction)[0].detach().float().cpu().numpy()
                    chunk = normalizer.denormalize(normalized_chunk, key="action")
                    timings[steps].append(latency_ms)
                    records.append({
                        "sequence_index": int(sample_index),
                        "episode_index": int(sample["episode_index"]),
                        "steps": int(steps), "repeat": int(repeat),
                        "seed": int(seed), "latency_ms": float(latency_ms),
                    })
                    outputs[steps][sample_index] = chunk
            if device.type == "cuda":
                cuda_peak_allocated[steps] = int(torch.cuda.max_memory_allocated(device))
                cuda_peak_reserved[steps] = int(torch.cuda.max_memory_reserved(device))
            else:
                cuda_peak_allocated[steps] = 0
                cuda_peak_reserved[steps] = 0
            print("INFERENCE_BENCH_PROGRESS " + json.dumps({
                "steps": steps, "latency_ms": percentile_summary(timings[steps]),
                "samples": len(windows),
            }), flush=True)
    except BaseException as error:
        import traceback
        (options.output / "failure.json").write_text(json.dumps({
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }, indent=2), encoding="utf-8")
        raise
    finally:
        resource_summary = resource_monitor.stop()

    reference = int(options.reference_steps)
    reference_outputs = outputs[reference]
    source_configuration = dataset.manifest["source_configuration"]
    joint_names = source_configuration["joint_names"]
    joint_units = source_configuration["joint_units"]
    action_deviation = {}
    for steps in steps_list:
        if steps == reference:
            action_deviation[str(steps)] = {
                "reference_steps": reference,
                "joint_differences": {
                    name: {"unit": unit, "mean_absolute_difference": 0.0,
                           "root_mean_square_difference": 0.0, "max_absolute_difference": 0.0}
                    for name, unit in zip(joint_names, joint_units)
                },
            }
            continue
        differences = np.stack([
            outputs[steps][sample["sequence_index"]] -
            reference_outputs[sample["sequence_index"]]
            for sample in windows
        ], axis=0)
        action_deviation[str(steps)] = {
            "reference_steps": reference,
            "joint_differences": {
                name: {
                    "unit": unit,
                    "mean_absolute_difference": float(np.mean(np.abs(differences[..., index]))),
                    "root_mean_square_difference": float(np.sqrt(np.mean(differences[..., index] ** 2))),
                    "max_absolute_difference": float(np.max(np.abs(differences[..., index]))),
                }
                for index, (name, unit) in enumerate(zip(joint_names, joint_units))
            },
        }
    report = {
        "schema": "autopicker-inference-benchmark-v1",
        "dataset_sha256": dataset_sha, "checkpoint_sha256": checkpoint_sha,
        "checkpoint_training_git_revision": metadata.get("git_revision"),
        "checkpoint_training_git_dirty": metadata.get("git_dirty"),
        "split": options.split, "seed": options.seed, "selected_episode_indices": indices,
        "sample_count": len(windows), "sample_sequence_indices": [w["sequence_index"] for w in windows],
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "steps": steps_list, "reference_steps": reference,
        "latency_ms": {str(steps): percentile_summary(timings[steps]) for steps in steps_list},
        "speedup_vs_reference_p50": {
            str(steps): float(percentile_summary(timings[reference])["p50"] /
                              percentile_summary(timings[steps])["p50"])
            for steps in steps_list
        },
        "action_deviation_vs_reference_physical_units": action_deviation,
        "action_features": [
            {"index": index, "name": name, "unit": unit}
            for index, (name, unit) in enumerate(zip(joint_names, joint_units))
        ],
        "cuda_peak_allocated_mb": {str(k): v / 1048576 for k, v in cuda_peak_allocated.items()},
        "cuda_peak_reserved_mb": {str(k): v / 1048576 for k, v in cuda_peak_reserved.items()},
        "resource_monitor": resource_summary,
        "run_manifest": "run_manifest.json", "run_manifest_sha256": manifest_sha,
        "measurement_scope": (
            "GPU/host policy forward-pass timing only; excludes worker IPC, "
            "Isaac Sim stepping, camera acquisition, and ROS 2 execution"),
        "training_scope": metadata.get("scope"),
    }
    (options.output / "metrics.jsonl").write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in records), encoding="utf-8")
    (options.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("AUTOPICKER_INFERENCE_BENCHMARK_PASS " + json.dumps({
        "latency_ms": report["latency_ms"],
        "action_deviation_vs_reference_physical_units": report["action_deviation_vs_reference_physical_units"],
    }), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--reference-steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args()
    if options.max_samples < 1 or options.warmup < 0 or options.repeats < 1:
        parser.error("Require max-samples >= 1, warmup >= 0 and repeats >= 1")
    benchmark(options)


if __name__ == "__main__":
    main()
