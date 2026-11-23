"""Lockstep JSON inference and endpoint action interpolation without Isaac imports."""
from collections import deque
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import threading
import time

import numpy as np


class ActionLoop:
    def __init__(self, predict, dt, lower, upper):
        self.predict, self.dt = predict, float(dt)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.history = deque(maxlen=2)
        self.requests = []

    def reset(self, generation, initial_command):
        self.generation = generation
        self.history.clear()
        self.requests.clear()
        self.anchor = np.asarray(initial_command, dtype=float).copy()
        self.actions = None

    def command(self, step, observation, current_command, physics_dt):
        stride = round(self.dt / physics_dt)
        if stride < 1 or abs(stride * physics_dt - self.dt) > 1e-8:
            raise ValueError('Policy dt must be a multiple of physics dt')
        if step % stride == 0:
            observation = np.asarray(observation, dtype=float)
            if observation.shape != (15,) or not np.isfinite(observation).all():
                raise ValueError('Invalid live observation')
            self.history.append(observation.copy())
            if len(self.history) == 1:
                self.history.append(observation.copy())
            if step % (2 * stride) == 0:
                started = time.monotonic()
                observations = np.stack(self.history)
                actions = np.asarray(self.predict(observations, self.generation, step * physics_dt), dtype=float)
                if (actions.shape != (2, 4) or not np.isfinite(actions).all()
                        or np.any(actions < self.lower - 1e-7) or np.any(actions > self.upper + 1e-7)):
                    raise ValueError('Policy returned invalid or out-of-limit actions')
                self.anchor = (np.asarray(current_command).copy() if self.actions is None
                               else self.actions[-1].copy())
                self.actions = np.clip(actions, self.lower, self.upper)
                self.chunk_start = step
                self.requests.append({'generation': self.generation, 'time_s': step * physics_dt,
                                      'observation': observations.tolist(), 'action': self.actions.tolist(),
                                      'latency_s': time.monotonic() - started})
        progress = (step - self.chunk_start) / stride
        segment = min(int(progress), 1)
        start = self.anchor if segment == 0 else self.actions[0]
        return start + (progress - segment) * (self.actions[segment] - start)


class PolicyClient:
    def __init__(self, python, checkpoint, upstream, seed, log_path, timeout=60.):
        self.timeout = timeout
        self.responses = Queue()
        self.sequence = 0
        self.log = Path(log_path).open('w', encoding='utf-8')
        # Isaac exports its own PYTHONHOME/PYTHONPATH (Python 3.11). Isolated
        # mode keeps the selected training interpreter's stdlib/site packages.
        self.process = subprocess.Popen([str(python), '-I', str(Path(__file__).with_name('serve.py')),
                                         '--checkpoint', str(checkpoint), '--upstream', str(upstream),
                                         '--seed', str(seed)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.ready = self._receive()
            if self.ready.get('status') != 'ready':
                raise RuntimeError('Policy process did not initialize')
        except Exception:
            self.close()
            raise

    def _receive(self):
        try:
            line = self.responses.get(timeout=self.timeout)
        except Empty as error:
            raise RuntimeError('Policy inference timed out') from error
        if line is None:
            raise RuntimeError('Policy process exited; inspect inference log')
        return json.loads(line)

    def predict(self, observation, generation, timestamp):
        self.sequence += 1
        request = {'request_id': self.sequence, 'generation': generation,
                   'time_s': timestamp, 'obs': observation.tolist()}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
        self.process.stdin.flush()
        response = self._receive()
        if any(response.get(key) != request[key] for key in ('request_id', 'generation', 'time_s')):
            raise RuntimeError('Stale or mismatched policy response')
        return response['action']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()
