# M4 dynamic recovery implementation plan

**Goal:** Compare six corrections from measured failed-policy states against the six static-reset recovery demonstrations in M3, with the original train/val grouping and success criteria unchanged.

**Architecture:** A separate dynamic_recovery module extracts the state at fixed 2 s and 6 s from each of the three M3 train rollouts. Restore all measured joint positions, active-arm measured velocities, and previous applied executor position/velocity; inactive velocities are explicitly zero because legacy raw episodes do not record them. No settling after restoration. This is an articulated-state reset, not a full PhysX solver checkpoint. Expert control records a new successful segment; learned evaluation remains free of expert IK. Freeze parent normalizer, seed 0, 3000 steps, batch 64, physical action limits and existing diffusion configuration. Replace the six static corrections with six dynamic corrections rather than combining both pools.

**Tech stack:** NumPy/unit tests, Isaac Sim 5.1, existing Zarr/PyTorch training environment.

The user's continuation accepts the dynamic expert-correction direction described in M3. This is the next bounded experiment on the existing research branch; no further approval gate is needed.

- [x] Test and implement state extraction/validation in research/autopicker_dp/dynamic_recovery.py. Reject held-out sources, changed dataset/checkpoint/raw hashes, attached states, invalid times and indices. `state_from_policy(source, time_s, indices)` returns q_all, active qdot, preceding executor command and velocity. Test that source row r uses executor row r-1 rather than the not-yet-applied command at r.
- [x] Implement `restore_dynamic_state(env, state)` with finite/limit checks, explicit zero inactive velocities, no physics step, verified first q/qdot and executor continuity. Keep ordinary environment reset unchanged. Add dynamic collection CLI with six fixed trials, per-trial failure retention and a complete summary. Test extraction, no-settle restoration and invalid-state failures before any mutation.
- [x] Add dynamic provenance validation to augmentation.py; static recovery remains accepted without changed windows. Check source identity, exact requested state and first raw measured q/qdot. Extend converted replay to restore declared dynamic initial state before replay, with validation. Test forged source/state and boundary behavior.
- [x] Run tests, collect all six expert corrections with 35 s budget and 0.2 s terminal hold; only export after 6/6 success. Convert to m4_dynamic_lowdim_v1 and require 6/6 converted replay under original 3 mm/1 degree capture and 25 mm collect criteria.
- [x] Train m4_train_dynamic_v1 using parent lowdim_v2 plus dynamic recovery export; freeze normalizer and budget. Require exact checkpoint reload; run offline diagnostics and five learned closed-loop trials on train [1,2,4] and val [0,5]. Compare against immutable M3 recorded trials, preserving failures.
- [x] Add validate_m4.py to bind source trajectories, restoration state, six expert/replay attempts, paired training settings and five policy attempts. Report the actual outcome and limitations, update README, independently review, commit and push research/diffusion-policy. Keep datasets/weights/logs local.

Validation: 65 tests passed; expert corrections 6/6 and converted replay 6/6. M4 learned train 0/3 and val 0/2; all failures retained. Independent final audit reproduced the report exactly. The fixed-budget replacement did not improve collection success.
