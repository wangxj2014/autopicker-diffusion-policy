# M3 recovery-data ablation

**Goal:** Measure whether six audited static recovery demonstrations improve the learned policy relative to the M2 train-only physical-normalization baseline.

**Design:** Use the original immutable lowdim_v2 as the primary dataset and evaluation identity. Append recovery episodes only to the training sequence pool, preserving episode boundaries and the original train/val/test assignment. Keep the entire M2 train-only normalizer unchanged, including observation statistics. Store both dataset manifests/hashes and the sampling rule in the checkpoint. This is uniform sampling over valid window anchors in the combined pool, not a balanced 50/50 mixture.

**Protocol:** seed 0, 3000 steps, batch 64, physical action normalization, same U-Net/EMA/DDIM20, 2 observed / 16 predicted / 2 executed steps, 10 Hz and 480 Hz physics. Capture 3 mm / 1 degree, collection 25 mm, 35 s timeout remain fixed. Original train-only baseline: m2_train_physical_v1. New model: m3_train_recovery_v1. Evaluate first three original train episodes (three distinct targets) for both models; evaluate the same two original validation episodes for the new model and compare to the recorded M2 baseline. Do not add test closed-loop trials or claim independent test generalization.

**Alternatives considered:** Copying raw episodes into a new merged export complicates evaluation identity and risks refitting normalization. A weighted mixture changes sample exposure beyond the simple augmentation question. Appending audited windows while freezing the original normalizer isolates the accepted experiment with fewer changes.

## Implementation and verification

- [x] Test and implement augmentation.py: reject recovery targets outside parent train, altered scene/control configuration, mismatched parent hashes, duplicate/failed sources, and invalid source/reset provenance. Append arrays and episode ends without crossing episode boundaries.
- [x] Add optional --recovery-dataset to train.py, limited to train + physical. Keep baseline behavior unchanged, record both dataset identities and uniform-window sampling in checkpoint/report. Test selected windows, frozen normalization and actual checkpoint metadata roundtrip.
- [x] Verify the full test suite, train the augmented model, require checkpoint reload equality, and run teacher-forced diagnostics on the original dataset.
- [x] Sequentially run three baseline train targets, three augmented train targets and two augmented validation episodes in Isaac. Retain all failures and original criteria; no IK in learned control.
- [x] Audit dataset lineage, paired model/config/normalizer, exact target/reset matching and all denominators; document results in docs/research/2026-11-23-m3-validation.{md,json}, with reproducible commands in research/README.md.
- [x] Request independent code review, fix findings, verify tests/artifacts, commit and push research/diffusion-policy. Leave raw datasets/weights local and original scene unchanged.

The user's continuation accepts the previously proposed recovery-data comparison. No additional design approval is needed. Dynamic recovery and training-budget changes are separate later experiments.
