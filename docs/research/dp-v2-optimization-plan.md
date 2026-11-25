# Diffusion Policy V2 optimization branch

## Scope of this first code change

This branch starts with a deliberately small, ablation-friendly change: optional
train-only Gaussian observation corruption on the four measured joint-velocity
channels (`qdot`) of the existing 15-dimensional low-dimensional observation.

The implementation is in `research/autopicker_dp/train.py`:

- CLI: `--qdot-noise-std-rad-s FLOAT`; default `0.0` preserves the previous
  training behavior.
- The requested physical standard deviation (rad/s) is converted to normalized
  units using the already-fitted training observation normalizer.
- Independent Gaussian noise is applied only to normalized observation channels
  `[..., 4:8]`. Joint positions, goal, receiver, and attachment state are not
  corrupted.
- Validation, test, offline evaluation, and rollout observations remain clean.
- The exact augmentation parameters and normalized standard deviations are
  recorded in checkpoint metadata.

This is a robustness regularizer, not a claim of improved picking success and
not a replacement for recovery demonstrations. It does not change the model
architecture, action labels, scheduler, inference loop, or EtherCAT control path.

## Suggested controlled experiment

Use the same dataset, split, seed, model, update budget, and action normalization
for all runs. Create a separate output directory for every run.

1. Baseline: `--qdot-noise-std-rad-s 0.0`.
2. Mild corruption: `--qdot-noise-std-rad-s 0.01`.
3. Moderate corruption: `--qdot-noise-std-rad-s 0.03`.
4. Stronger corruption: `--qdot-noise-std-rad-s 0.05`.

These are proposed sweep values, not validated optima. Repeat promising settings
with multiple training seeds. Compare physical-unit offline errors, clean
held-out closed-loop success, inference latency, and failure categories. Keep
pure-policy results separate from hybrid/IK-assisted results. Do not select a
setting from training loss alone.

Example (run once per value with a unique output directory):

```bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py \
  --dataset datasets/autopicker/lowdim_v2 \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp_v2_qdot_noise_001_seed0 \
  --split train --action-normalization physical \
  --steps 3000 --batch-size 64 --seed 0 \
  --qdot-noise-std-rad-s 0.01
```

## Tests

Run the focused augmentation test and then the complete existing suite:

```bash
python -m unittest research.tests.test_observation_augmentation -v
AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream \
  work/autopicker_train_env/bin/python -m unittest discover -s research/tests -v
```

The first command requires PyTorch. The full suite uses the pinned upstream
Diffusion Policy checkout and the repository's isolated training environment.
Isaac Sim closed-loop evaluation is not run by a GitHub source edit and must be
executed on the configured simulation machine.

## Preserved constraints

- The source branch `research/diffusion-policy` and `main` are unchanged.
- The low-dimensional 15D observation / 4D joint-action contract is unchanged.
- Training normalizer provenance remains train-only; no held-out target is used
  to fit normalization.
- The model remains a high-level action-chunk policy. It must not be connected
  directly to the 4 ms EtherCAT servo loop.
- No claim of sim-to-real readiness or orchard-level success is made.
