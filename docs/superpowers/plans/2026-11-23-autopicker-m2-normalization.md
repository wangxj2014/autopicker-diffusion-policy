# M2 action normalization and recovery pilot

> Execute in the existing research branch with test-driven changes and measured Isaac rollouts. The user has authorized continued research; no additional design approval is needed.

**Goal:** Remove the training-envelope restriction on legal joint commands, measure its isolated effect, and collect a bounded recovery pilot if closed-loop failures persist.

**Architecture:** Keep the frozen lowdim_v2 data export intact. Copy its selected-episode observation normalizer and optionally replace only the action affine transform with physical joint limits from source_configuration. Save the exact choice in checkpoints. Recovery collection is a separate expert-only workflow; learned evaluation never calls IK.

**Frozen protocol:** seed 0, 3000 optimizer steps, batch 64, same U-Net/EMA/DDIM20 with clipping, observation/action horizons 2/16, execute 2, 10 Hz data and 480 Hz physics. Keep capture 3 mm / 1 degree, collect 25 mm, timeout 35 s. Compare five seen-target all-pilot diagnostics and two validation repeats to M1. Treat the previously inspected test group as pilot data, not a pristine publication test set.

## Tasks

- [x] Add failing normalizer tests in research/tests/test_dataset.py: legal endpoints map to ±1, unseen legal actions remain representable, observation statistics and original object stay unchanged, JSON roundtrip, invalid bounds fail.
- [x] Implement Normalizer.with_action_limits(lower, upper) in research/autopicker_dp/dataset.py and run focused tests.
- [x] Add --action-normalization {data,physical} to train.py (default data for reproducibility), selection helper and checkpoint metadata; test real checkpoint mapping roundtrip in test_policy.py.
- [x] Run the full unittest suite in work/autopicker_train_env with AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream.
- [x] Train m2_overfit_physical_v1 and m2_train_physical_v1, each 3000 steps; require TRAIN_AND_RELOAD_PASS. Evaluate offline and sequentially run five seen-target and two validation Isaac trials. Retain failures and full denominators.
- [x] If failures persist, implement and validate deterministic train-target-only recovery collection from perturbed states, preserving physical execution, reset provenance and failure accounting. Collect a bounded pilot and replay it before recommending training on it.
- [x] Record hashes, denominators, precision and limitations in docs/research/2026-11-23-m2-validation.{md,json}; update research/README.md with reproducible commands.
- [x] Review changes and verify tests and artifact audit. Independent review has no remaining blockers; 52 tests pass, paired normalization audit passes, recovery collection and converted replay each collect 6/6.

Delivery: commit and push research/diffusion-policy after these checks. Keep datasets, weights and raw runs local; leave main and demo assets intact. Final delivery status is recorded in the task response and Git history.
