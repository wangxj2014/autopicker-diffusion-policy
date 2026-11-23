# M5 direct action prediction implementation plan

**Goal:** Reduce diffusion action uncertainty and obtain measured learned collection without expert IK or relaxed thresholds.

**Design and evidence:** M3, which remains stronger than M4 on the seen targets, produces mean seed standard deviations [0.01887 rad, 0.00909 m, 0.00761 m, 0.03855 rad] on 103 fixed parent-train observation windows and eight noise seeds. This is actionable evidence of noisy commands, not proof of the sole failure cause. Keep M3's 820 + 685 training windows, full parent normalizer, physical action bounds, 3000 updates, seed 0, network, DDIM20 and action executor. Compare epsilon prediction to clean-action (`sample` / x0) prediction, with scheduler and loss targets consistent. All existing checkpoints default to epsilon and retain their behavior.

Alternatives considered: state-relative actions change the representation and affine mapping; fixed-noise or sample averaging changes inference behavior and/or cost. Clean-action prediction is the smallest first intervention. User authorization to continue improving picking covers this bounded investigation and implementation; no repeated approval gate is needed.

**Architecture:** Add `prediction_type` to DiffusionPolicy configuration and training CLI. Loss uses noise for epsilon and clean normalized actions for sample; checkpoint loader checks configured objective matches saved scheduler. No inference-loop, data, reset, capture or collect changes. Use existing separate process for learned rollouts. The experiment is single-scene/oracle/single-seed development evidence.

**Execution:** Inline implementation and independent reviewer via requesting-code-review skill.

- [x] Add tests for clean-action loss target/masking, nonzero gradients, objective checkpoint round-trip, invalid objectives and incompatible saved schedulers; preserve legacy epsilon loading. Run RED before implementation.
- [x] Add backward-compatible objective option to policy.py and train.py. Run focused/full tests and train `m5_train_sample_v1` with static recovery data, 3000 steps. Validate reload and offline accuracy/noise on the same probes used for M3.
- [x] Run learned train [1,2,4] and val [0,5] trials using seed 0, 35 seconds, original 3 mm / 1 degree capture and 25 mm collection thresholds. Preserve every failure. No IK fallback in learned rollouts.
- [x] If the 3000-step model still lacks reliable seen-target collection and diagnostics show underfitting, run one separately labeled 30000-step budget experiment with all other conditions unchanged. Report it separately from the equal-budget objective comparison. Evaluate three seen targets and two validation episodes; do not search noise seeds or thresholds for successes.
- [x] Add M5 audit/report binding datasets, checkpoint objectives/budgets, paired baseline and all attempts. Diagnose remaining errors rather than infer causality from loss. Update commands/docs, independently review, verify, commit and push the existing research branch. Raw trajectories and weights remain local.

**Equal-budget outcome:** 0/3 seen-target and 0/2 validation capture/collection. Seen-target minimum distances 3.579/4.143/10.835 mm; fixed-probe horizontal/vertical biases remain 1.380/0.551 mm. This motivates the single preplanned 30000-step precision diagnostic without claiming underfitting is the sole cause. Same seed retraining does not enforce CUDA determinism; early training losses are not bitwise identical.

**Longer-budget outcome:** 0/3 seen-target and 0/2 validation capture/collection. Seen minimum distances 8.724/7.852/20.683 mm, validation 85.804/85.102 mm. Teacher-forced train horizontal/vertical MAE reached 0.119/0.073 mm, yet closed-loop collection remained absent. No further checkpoint, seed or threshold search was performed. Complete M5 artifact audit passed; 73 tests passed.

**Final verification:** Independent CPU audit exactly reproduced the committed JSON; reviewer found no remaining blocking issues. Full 73-test suite and diff whitespace checks passed. Documentation and raw trajectory figure preserve the negative collection outcome.
