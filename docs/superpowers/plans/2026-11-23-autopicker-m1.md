# Autopicker M1 Implementation Plan

> **For agentic workers:** Use the existing approved research design and execute the coupled data/training/rollout tasks in sequence; request independent code review before publishing.

**Goal:** Turn the audited M0 pilot into reproducible low-dimensional diffusion-policy training and a measured Isaac rollout, without interpreting pilot fitting as generalization.

**Architecture:** Ordinary Python converts raw episodes into fixed-rate oracle observations and next-node joint commands. An isolated PyTorch environment uses the pinned upstream ConditionalUnet1D and a diffusion scheduler. A separate inference process exchanges JSON observations/actions with Isaac in lockstep; only the existing executor writes robot targets.

**Tech Stack:** NumPy, Zarr v2, PyTorch 2.7.0+cu126, upstream Diffusion Policy commit `5ba07ac6661db573af695b419a7947ecb704690f`, diffusers.

## Frozen choices

- Keep the user-selected `research/diffusion-policy` branch. Baseline: `4edcb5e`; original scene/demo unchanged.
- Group by `(scene_sha256, target_path)`. With five pilot targets, sorted identities allocate three to train, one to validation, one to test. Duplicate trajectories stay in the same group. This split is only a pipeline pilot.
- Observation is `q(4), qdot(4), goal_base_m(3), receiver_base_m(3), attached(1)`. All spatial transforms use recorded USD row-vector convention. No phase/time/target identifier is a network input.
- At observation time `t`, label is the applied expert command at `t+0.1`. Use complete 0.1 s intervals only; preserve omitted final duration in metadata and terminal state separately. Do not interpolate observations across events.
- `horizon=16`, two observations, two executed actions. Sequence starts at `k-1`; current action starts at model index one. Boundary replication includes a validity mask; no cross-episode windows.
- Fit normalization on train only. A separately named all-pilot overfit run may use all ten episodes; never label its rollout held-out performance.
- Training and inference code/config/source revisions, data hashes, seeds, optimizer state, normalizer and scheduler go into each checkpoint. Fixed-seed reload must reproduce actions.
- First experiment has a finite budget. Report actual rollout successes/failures and latency, without adjusting the 3 mm/1° task gate or falling back to IK.

## Tasks

- [x] Add `dataset.py` and failing tests for next-node labels, transformed coordinates, terminal truncation, target grouping, train-only normalization and boundary masks. Run before and after implementation.
- [x] Add conversion CLI, immutable manifest/split metadata and Zarr v2 export. Initial `pilot_rgbd_v2` conversion exposed lost terminal collect commands. Recollect with 0.2 s measured post-collect hold into `pilot_lowdim_tail_v1`; final export `lowdim_v2` has 1,326 rows and audits successfully.
- [x] Add isolated training dependencies, pinned upstream validation, `policy.py` and `train.py`. Test noise loss/backward, `(B,16,4)` prediction, `[1:3]` action extraction and fixed-seed checkpoint reload. Both all-pilot and train-only runs completed 3,000 steps with exact reload outputs.
- [x] Add lockstep policy inference and an Isaac rollout entry. Converted replay: 10/10; learned all-pilot diagnostic: 0/5; train-only validation: 0/2. All seven failures retained; no expert fallback or gate relaxation.
- [x] Review code, run focused regression checks, write an honest report with experiment limits, update research documentation, and prepare the research branch for publication. Publishing follows the verified checks and is recorded in Git history.

## Acceptance

Data and checkpoint audits must pass before rollout. A training loss decrease alone is not task success. M1 research acceptance remains open if the learned policy cannot collect fruit; preserve failures and state the next technical issue. No visual policy, multi-arm policy, scene randomization or paper success claim belongs to this pilot.

Result: data/interface audit passed; robust learned collection remains open. The first learned experiment reveals train-envelope action clipping and closed-loop state drift. See `docs/research/2026-11-23-m1-validation.md` for evidence and the next bounded experiments. Independent reviewer was unavailable due a provider error; main-agent review and automated/runtime checks were completed.
