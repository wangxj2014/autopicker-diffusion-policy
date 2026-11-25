# M7 Corridor Handoff Implementation Plan

> **For agentic workers:** Use executing-plans inline; request an independent review at the final checkpoint. Existing user authorization covers implementation and the bounded experiment.

**Goal:** Reduce repeated learned approach/retreat before precision control using a measured pose gate and paired episode randomness.

**Architecture:** Extend the existing hybrid with an entry hook and a corridor subclass. Add an opt-in episode seed protocol to client/server/rollout. Reuse stage/kinematic audits with independently calculated entry masks, then freeze and compare two controllers.

**Tech Stack:** Python, unittest, NumPy, existing IK/JointExecutor, Isaac Sim 5.1, frozen PyTorch Diffusion Policy.

## Task 1: Independent episode seeds

- [x] Add `research/tests/test_episode_seeds.py`: create protocol clients with in-memory streams, reset to source 4 after different previous request counts, and require identical seeds despite different request IDs. Require source 5 and trial seed 1 to differ; reject a response with changed noise seed. Legacy requests must retain their original fields.
- [x] Run `work/autopicker_train_env/bin/python -m unittest discover -s research/tests -p test_episode_seeds.py -v`, observe missing-feature failure.
- [x] Implement `episode_noise_seed(base_seed, source_index, local_index)` and `PolicyClient.reset_episode(source_index)`. Use:

```python
payload = f'autopicker-episode-v1:{base_seed}:{source_index}:{local_index}'.encode()
seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
```

- [x] Echo optional seed fields in `serve.py`; bind response equality to all identity fields. Log them from LivePolicy. Add `--seed-mode {global,episode}`, default global, to rollout metadata and summary, reset before each episode in episode mode. Run focused inference and rollout tests.

## Task 2: Measured corridor entry

- [x] Extend `test_hybrid.py` with aligned 10 cm entry, lateral/angle/behind rejection, 30 mm fallback, latched precision and capture-gated return. Observe RED before implementing.
- [x] Extract `_precision_entry(env)` from `HybridPolicy.command` without changing M6 behavior. Add `CorridorHybridPolicy` and frozen `CORRIDOR_CONFIG`. Outside the original sphere use:

```python
delta = env.entry['world'] - env.tip_pose()[:3, 3]
axial = float(delta @ env.direction)
lateral = float(np.linalg.norm(delta - axial * env.direction))
eligible = distance <= .12 and axial >= 0 and lateral <= .02 and angle <= 5.
```

- [x] Add explicit rollout `hybrid-corridor` mode and record configuration/scope. Run focused and full tests. Do not modify execution limits, timeout, capture, collect, scene or trained weights.

## Task 3: Fixed 20-trial evaluation and independent audit

- [ ] Run sequential Isaac rollouts for mode hybrid/hybrid-corridor, seeds 0/1, train first 3/val first 2, with `--seed-mode episode --timeout 35 --headless`; save to `runs/autopicker/m7_<mode>_<split>_s<seed>`. Check completion marker, summary and traceback for each run.
- [x] Add `research/tests/test_m7_audit.py` negative cases for forged seed, changed source, premature/late entry, modified angle, and unequal paired prefixes. Add `validate_m7.py` to bind all 20 attempts, dataset/model/source hashes/reset/seed protocol; use raw FK for orientation and entry geometry. Extend M6 stage audit with explicit optional configuration/entry-mask inputs, retaining default behavior. Reconstruct commands and real event gates, compare common raw/request prefixes before handoff.
- [ ] Run full suite and `work/autopicker_train_env/bin/python research/tests/validate_m7.py --output docs/research/2026-11-25-m7-validation.json`. Write matching Markdown, static trajectory plot and README launch commands. Preserve failures and note validation is not an untouched test set.
- [x] Request independent review; fix findings, verify evidence, commit and push research/diffusion-policy. Do not merge main.


## Execution update: external GPU blocker

The first launch stopped before Isaac/any episode: attempted=0. CUDA cuInit returned 999; kernel logs from before this experiment record NVIDIA Xid 154 / Node Reboot Required. Preserve the startup failure at `runs/autopicker/m7_hybrid_train_s0_startup_failure_20261125`. No new success rate or 20-trial acceptance is reported.

- [x] Add and test `research/run_m7.py`: default prints the fixed matrix; execution checks CUDA before creating outputs, runs one Isaac group at a time, refuses partial/changed-source reuse, and audits all 20 attempts on completion.
- [x] Add and run `research/diagnostics/m7_readiness.py`: audit five archived M6 trajectories, calculate hypothetical first entry and terminal IK feasibility, and verify the real serve subprocess on CPU. Do not interpret hypothetical entries as executed trajectories.
- [x] Full CPU suite: 98 PASS. M7 full artifact auditor refuses missing experiments and writes no passing JSON.
- [x] Write `docs/research/2026-11-25-m7-readiness.{md,json}`, `2026-11-25-m7-status.json` and README instructions, with explicit experimental / 0 of 20 status. The planned closed-loop result plot remains deferred with the experiment.
- [x] Final independent review and research-branch commit/push. The 20-trial experiment remains pending GPU recovery, regardless of code-delivery status.

- [x] Independent review fixes: add a successful-exit receipt binding process log and summary; require it for resume and final audit. Preserve all inference attempt identities and global start/end spans in successful and failed episode records. New regression cases first failed, then full suite passed with 98 tests.
