# M7: pose-qualified early precision handoff

The user authorized continued optimization. M6 archived trajectories show that all three seen targets first satisfy a 12 cm distance, 2 cm lateral error and 5 degree orientation gate at 3.03–3.13 s, well before the 30 mm handoff. Validation Apple_036 does not satisfy that gate: its closest approach has 73–76 mm lateral error and 15–17 degrees orientation error. These are exploratory observations, not new evaluation results.

## Design and alternatives

Add an explicit `hybrid-corridor` mode. Keep M6's 30 mm entry as a fallback. Outside it, require measured tip distance <= 0.12 m, nonnegative approach-axis distance, lateral error <= 0.02 m and orientation error <= 5 degrees. The direction is the existing upward 30 degree grasp direction. Once qualified, use the existing latched IK precision and same-frame capture-gated return. There is no time-based fallback, no increased capture tolerance, and no return before attachment. Joint interpolation targets the 30 degree final pose; this change does not establish a straight Cartesian path or collision avoidance.

An unconditional larger sphere would accept lateral/off-axis approaches. A new learned phase or velocity correction needs its own data/training study. This bounded iteration tests whether an earlier well-aligned handoff removes wasted approach time while preserving the original hybrid control after entry. It deliberately does not promise to solve the held-out lateral error.

## Paired randomness

Add opt-in `--seed-mode episode` (legacy global sequence remains default). Derive each inference noise seed from SHA256 of `autopicker-episode-v1:<trial seed>:<source episode index>:<local request index>`, first four bytes big endian. Reset local request index to zero per source episode; retain globally increasing request IDs for stale-response checks. Echo and log source index, local index and noise seed, and reject mismatches. This makes noise independent of previous episode duration, controller handoff or episode ordering.

## Frozen experiment

Before running new simulations, freeze weights at M5 sample 30000 checkpoint SHA256 d0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a. Use parent lowdim_v2, train source indices [1,2,4], validation [0,5], trial seeds [0,1], both `hybrid` and `hybrid-corridor`: 20 attempts total. Source [0,5] repeats one validation target; validation has already informed diagnosis and is not an untouched test set. This round adds no test-split evaluation; historical M1–M5 offline test results have already been used for development diagnostics.

Same scene/reset, 35 s timeout, 3 mm/1 degree capture, 25 mm collect, speed and acceleration limits, no training or other GPU workload during rollouts. Require completed summary and completion marker with no traceback. Preserve every failure in the denominator. Compare capture, collection, handoff time, collection time (also 35 s capped time across all attempts), raw prefixes until first intervention, per-request seeds, actual event/attachment transitions, FK and reconstructed commands. Do not compare new episode-seeded results as if paired with old M6 global seeds.

## Scope and delivery

Oracle low-dimensional state, fixed scene, one arm, visual attachment. Report hybrid success separately from learned capability. No changes to showcase USD, cameras or demo. Commit code, small diagnostic/result reports and plot; retain weights and raw trials locally. Historical M6 remains reproducible at commit 29df26c because its validator binds the source version used for its rollouts.


## Execution status

2026-11-25: implementation and CPU readiness checks completed; new closed-loop attempts 0/20. A pre-existing NVIDIA Xid 154 requires node reboot. The planned matrix and thresholds remain fixed. Readiness documentation and a guarded runner are delivered separately from the unavailable experiment result; no effectiveness claim is made.
