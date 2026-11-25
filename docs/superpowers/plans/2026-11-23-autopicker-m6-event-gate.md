# M6 Event Gate Implementation Plan

> Execute inline with the executing-plans workflow. Existing user authorization covers the bounded experiment; use an independent code reviewer before completion.

**Goal:** Stop learned approach/retreat cycling after entering the precision region, and permit return only after a measured capture.

**Architecture:** A separate hybrid action provider wraps the unchanged LivePolicy. It latches precision IK and then a capture-triggered release trajectory. Every classical intervention is labeled and auditable; pure-policy rollouts remain available.

**Tech Stack:** Python, NumPy, existing JointChain/SciPy IK, Isaac Sim 5.1, unittest.

## Task 1: Event-gated provider

- [x] Add `research/tests/test_hybrid.py`. Use a fake environment with a real Episode event list and a counting coarse policy. Assert `command(env)` initially delegates, at 30 mm starts at executor.position, remains in precision after distance increases, and never switches on elapsed time alone. Set an actual capture event and attached/phase, then assert return begins at the current executor position and freezes the measured hor in its goal. Reject invalid IK, unattached return, fabricated attachment and stale reset generation.
- [x] Run `work/autopicker_train_env/bin/python -m unittest discover -s research/tests -p test_hybrid.py -v`; expect failure because HybridPolicy is missing.
- [x] Create `research/autopicker_dp/hybrid.py` with `HybridPolicy(policy, env)` and `command(env)`. Reuse `motion_duration`, `quintic_blend`, `capture_errors` and `vector`; do not modify the shared executor. The decisive branch is:

```python
if env.arm['attached'] and self.phase != 'return':
    # Require same-frame measured capture; begin from env.executor.position.
    goal = env.arm['release_goal'].copy()
    goal[1] = env.positions()[1]
elif self.phase == 'policy' and distance <= .03:
    goal = env.arm['chain'].inverse_approach(env.entry['world'], env.positions(), 30.)
    # Validate FK residual, latch goal until actual capture.
```

- [x] Rerun focused tests; expect PASS.

## Task 2: Explicit rollout mode

- [x] Extend `research/autopicker_dp/rollout.py` mode choices to `converted-replay`, `policy`, `hybrid`; require the same checkpoint arguments for both learned modes. Instantiate LivePolicy first and wrap it only for hybrid. Store controller kind/config in summary and episode metadata, with explicit oracle IK scope. Preserve every attempt.
- [x] Run the existing rollout/inference tests and full suite:
  `AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream work/autopicker_train_env/bin/python -m unittest discover -s research/tests -v`.

## Task 3: Frozen closed-loop experiment and audit

- [x] Run Isaac `research/autopicker_dp/rollout.py --mode hybrid --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m5_train_sample_long_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --split train --max-episodes 3 --seed 0 --timeout 35 --output runs/autopicker/m6_hybrid_train_v1 --headless`.
- [x] Run the same command sequentially for `--split val --max-episodes 2 --output runs/autopicker/m6_hybrid_val_v1`. Require completion marker, completed summary and absence of traceback; successful process exit alone is insufficient.
- [x] Add `research/tests/validate_m6.py` and negative tests in `test_m6_audit.py` for premature return, fake capture, continuing inference after handoff and changed protocol. Reconstruct expected applied commands through JointExecutor from recorded stage trajectories; verify reset/data/model identity and state event boundaries. Reuse baseline validator for M5 and generic rollout report for counts.
- [x] Create `docs/research/2026-11-23-m6-validation.{json,md}` and trajectory PNG showing distance, attachment and stage boundaries; report zeroes and limitations explicitly. Update both README files with hybrid launch command and honest scope.
- [x] Independent review, fix findings, run full tests and artifact audit, commit and push `research/diffusion-policy`. Keep large run artifacts local.

## Execution evidence

- Added and executed behavioral regression tests before implementation; initial missing HybridPolicy/validator failures recorded, then green.
- Frozen five-trial experiment: train capture 3/3, collect 2/3; validation capture/collect 0/2. Third train trial captured at 32.6333 s and timed out during return at 35 s.
- Full suite: 82 tests passed. Artifact audit: AUTOPICKER_M6_AUDIT_PASS.
- Independent review caught missing FK validation in the artifact audit. Added negative tests and independent raw joint/goal kinematics checks; real artifacts still pass. No controller or run record was changed after simulation.
