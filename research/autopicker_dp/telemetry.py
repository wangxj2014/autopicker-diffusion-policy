"""Single-host experiment telemetry using only the Python standard library.

Samples host CPU/RAM and NVIDIA GPU utilization/memory while a run is active.
The process-tree fields include the current process and its children (for example,
Isaac Sim plus the isolated inference worker).
"""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time


def _run(command, timeout=3.0):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode == 0:
            return result.stdout.strip(), None
        return "", result.stderr.strip() or f"exit status {result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as error:
        return "", str(error)


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _git_snapshot(root):
    revision, revision_error = _run(["git", "rev-parse", "HEAD"], timeout=2)
    branch, branch_error = _run(["git", "branch", "--show-current"], timeout=2)
    status, status_error = _run(["git", "status", "--porcelain"], timeout=2)
    return {
        "revision": revision or None,
        "branch": branch or None,
        "dirty": bool(status) if status_error is None else None,
        "git_errors": [error for error in (revision_error, branch_error, status_error) if error],
    }


def _os_release():
    result = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value.strip().strip('"')
    except OSError:
        pass
    return result


def _gpu_static():
    output, error = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ])
    if error:
        return {"available": False, "error": error, "devices": []}
    import csv as csv_module
    devices = []
    for row in csv_module.reader(output.splitlines()):
        if not row:
            continue
        values = [item.strip() for item in row]
        if len(values) >= 6:
            devices.append({
                "index": values[0], "name": values[1], "uuid": values[2],
                "driver_version": values[3], "memory_total_mb": values[4],
                "pci_bus_id": values[5],
            })
    return {"available": bool(devices), "devices": devices}


def _memory_snapshot():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields:
                values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return {}
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    return {
        "ram_total_mb": round(total / 1048576, 2) if total else None,
        "ram_available_mb": round(available / 1048576, 2) if available else None,
        "ram_used_mb": round((total - available) / 1048576, 2) if total else None,
    }


def _cpu_counters():
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    except (OSError, ValueError, IndexError):
        return None


def _process_tree_snapshot(root_pid):
    """Return CPU seconds, RSS bytes and process count for root plus descendants."""
    records = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text()
            tail = raw[raw.rfind(")") + 2:].split()
            pid = int(item.name)
            parent = int(tail[1])
            cpu_ticks = int(tail[11]) + int(tail[12])
            rss_pages = int(tail[21])
            records[pid] = (parent, cpu_ticks, max(0, rss_pages) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in records.items():
            if pid not in selected and parent in selected:
                selected.add(pid)
                changed = True
    selected_records = [records[pid] for pid in selected if pid in records]
    ticks = os.sysconf("SC_CLK_TCK")
    return {
        "process_tree_cpu_s": sum(item[1] for item in selected_records) / ticks,
        "process_tree_rss_mb": sum(item[2] for item in selected_records) / 1048576,
        "process_tree_count": len(selected_records),
    }


def _gpu_dynamic():
    output, error = _run([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if error:
        return [], error
    import csv as csv_module
    devices = []
    for row in csv_module.reader(output.splitlines()):
        values = [item.strip() for item in row]
        if len(values) < 7:
            continue
        def number(value):
            try:
                return float(value) if value not in ("", "N/A", "[Not Supported]") else None
            except ValueError:
                return None
        devices.append({
            "gpu_index": values[0],
            "gpu_util_percent": number(values[1]),
            "gpu_memory_util_percent": number(values[2]),
            "gpu_memory_used_mb": number(values[3]),
            "gpu_memory_total_mb": number(values[4]),
            "gpu_temperature_c": number(values[5]),
            "gpu_power_w": number(values[6]),
        })
    return devices, None


def _gpu_processes():
    output, error = _run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if error:
        return [], error
    import csv as csv_module
    processes = []
    for row in csv_module.reader(output.splitlines()):
        values = [item.strip() for item in row]
        if len(values) >= 3:
            processes.append({"pid": values[0], "process_name": values[1], "used_gpu_memory_mb": values[2]})
    return processes, None


def write_run_manifest(output_dir, repo_root, run_type, configuration=None, input_paths=None, extra=None):
    """Write a compact, secret-free description of the run and its inputs."""
    output_dir = Path(output_dir)
    repo_root = Path(repo_root).resolve()
    inputs = {}
    for name, raw_path in (input_paths or {}).items():
        path = Path(raw_path).expanduser().resolve()
        try:
            display_path = str(path.relative_to(repo_root))
        except ValueError:
            display_path = str(path)
        item = {"path": display_path, "exists": path.exists()}
        if path.is_file():
            item["size_bytes"] = path.stat().st_size
            item["sha256"] = _sha256(path)
        inputs[name] = item
    manifest = {
        "schema": "autopicker-run-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": str(run_type),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "os_release": _os_release(),
            "architecture": platform.machine(),
            "cpu_count_logical": os.cpu_count(),
            **_memory_snapshot(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "gpu": _gpu_static(),
        "git": _git_snapshot(repo_root),
        "configuration": _jsonable(configuration or {}),
        "inputs": inputs,
        "extra": _jsonable(extra or {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


class ResourceMonitor:
    """One-Hz host/GPU monitor; records the current process tree and all visible GPUs."""

    FIELDS = (
        "timestamp_utc", "elapsed_s", "system_cpu_percent", "process_tree_cpu_percent",
        "process_tree_rss_mb", "process_tree_count", "load_1m", "ram_total_mb",
        "ram_used_mb", "ram_available_mb", "gpu_index", "gpu_util_percent",
        "gpu_memory_util_percent", "gpu_memory_used_mb", "gpu_memory_total_mb",
        "gpu_temperature_c", "gpu_power_w", "gpu_processes_json",
    )

    def __init__(self, output_dir, interval_s=1.0, root_pid=None):
        if not isinstance(interval_s, (int, float)) or interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.output_dir = Path(output_dir)
        self.interval_s = float(interval_s)
        self.root_pid = int(root_pid or os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.rows = []
        self.errors = []
        self._previous_cpu = None
        self._previous_tree_cpu = None
        self._previous_time = None
        self._last_gpu_processes = []
        self._last_gpu_process_error = None
        self._sample_count = 0
        self._lock = threading.Lock()

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "system_metrics.csv"
        self.error_path = self.output_dir / "resource_monitor_errors.jsonl"
        self.stream = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._worker, name="autopicker-resource-monitor", daemon=True)
        self.thread.start()
        return self

    def _record_error(self, component, message):
        if not message:
            return
        item = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "component": component, "error": str(message)}
        self.errors.append(item)
        with self.error_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")

    def _sample(self):
        now = time.monotonic()
        elapsed = now - self.started
        cpu = _cpu_counters()
        system_cpu_percent = None
        if cpu is not None and self._previous_cpu is not None:
            total_delta = cpu[0] - self._previous_cpu[0]
            idle_delta = cpu[1] - self._previous_cpu[1]
            if total_delta > 0:
                system_cpu_percent = max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))
        self._previous_cpu = cpu
        tree = _process_tree_snapshot(self.root_pid)
        process_tree_cpu_percent = None
        if self._previous_tree_cpu is not None and self._previous_time is not None:
            duration = now - self._previous_time
            if duration > 0:
                process_tree_cpu_percent = max(0.0, 100.0 * (tree["process_tree_cpu_s"] -
                                                               self._previous_tree_cpu) / duration /
                                               max(1, os.cpu_count() or 1))
        self._previous_tree_cpu = tree["process_tree_cpu_s"]
        self._previous_time = now
        memory = _memory_snapshot()
        try:
            load_1m = float(Path("/proc/loadavg").read_text().split()[0])
        except (OSError, ValueError, IndexError):
            load_1m = None
        devices, gpu_error = _gpu_dynamic()
        self._record_error("nvidia-smi", gpu_error)
        if self._sample_count % 5 == 0:
            self._last_gpu_processes, self._last_gpu_process_error = _gpu_processes()
            self._record_error("nvidia-smi-processes", self._last_gpu_process_error)
        common = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 3),
            "system_cpu_percent": system_cpu_percent,
            "process_tree_cpu_percent": process_tree_cpu_percent,
            "process_tree_rss_mb": round(tree["process_tree_rss_mb"], 2),
            "process_tree_count": tree["process_tree_count"],
            "load_1m": load_1m,
            **memory,
            "gpu_processes_json": json.dumps(self._last_gpu_processes, ensure_ascii=False),
        }
        if devices:
            rows = [{**common, **device} for device in devices]
        else:
            rows = [{**common, "gpu_index": None, "gpu_util_percent": None,
                     "gpu_memory_util_percent": None, "gpu_memory_used_mb": None,
                     "gpu_memory_total_mb": None, "gpu_temperature_c": None, "gpu_power_w": None}]
        for row in rows:
            self.writer.writerow({key: row.get(key) for key in self.FIELDS})
        self.stream.flush()
        self.rows.extend(rows)
        self._sample_count += 1

    def _worker(self):
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                with self._lock:
                    self._sample()
            except Exception as error:
                self._record_error("sampler", error)
            deadline += self.interval_s
            self.stop_event.wait(max(0.0, deadline - time.monotonic()))

    def stop(self):
        if self.thread is None:
            return {"sample_count": 0, "errors": []}
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_s * 3))
        with self._lock:
            if not self.thread.is_alive():
                self.stream.close()
        numeric = {}
        for field in ("system_cpu_percent", "process_tree_cpu_percent", "process_tree_rss_mb",
                      "ram_used_mb", "gpu_util_percent", "gpu_memory_used_mb", "gpu_temperature_c",
                      "gpu_power_w"):
            values = [float(row[field]) for row in self.rows
                      if row.get(field) not in (None, "") and _is_number(row.get(field))]
            if values:
                numeric[field] = {"mean": round(sum(values) / len(values), 3),
                                  "max": round(max(values), 3)}
        return {
            "sample_count": self._sample_count,
            "interval_s": self.interval_s,
            "metrics_csv": self.csv_path.name,
            "errors_file": self.error_path.name if self.errors else None,
            "error_count": len(self.errors),
            "aggregates": numeric,
        }


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
