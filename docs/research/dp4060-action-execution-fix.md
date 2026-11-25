# DP4060 Action Execution Fix (M8)

## Purpose

This experiment isolates the closed-loop action handoff between the low-dimensional Diffusion Policy and the bounded joint executor. It is based on the 2026-11-25 10-step validation report, where both pilot episodes timed out and the recorded policy position targets frequently exceeded the per-100-ms velocity budget.

## Changes

1. **Replan from the current executor position.** At every policy replan, `ActionLoop` now anchors the new action segment at `current_command` (the bounded executor's current position). It no longer assumes the previous predicted endpoint has already been reached. The predicted action nodes remain unchanged and are still checked against joint position limits.
2. **Record 100-ms trajectory diagnostics.** Policy rollouts write `episode_N/trajectory.csv` with measured joint position/velocity, both predicted action nodes, interpolated requested command, executor position/velocity, tool and target world positions, tool-to-target distance, and executor limiting counters. The CSV is also described in the episode metadata. Hybrid controllers are unwrapped so the learned-policy phase can be diagnosed.
3. **Persist final run status reliably.** Rollout summaries are written atomically after each episode and at shutdown. The final report includes `run_status` and `finished_at_utc`; monitor shutdown errors are retained rather than preventing the summary write.
4. **Add a regression test.** The ActionLoop test now verifies that a new segment begins at the latest executor position when the previous predicted endpoint has not been reached.

## Validation

Run the Isaac-independent inference tests with the project's training Python:

```bash
python -m unittest discover -s research/tests -p 'test_inference.py' -v
```

The test suite is committed, but was not executed in the remote GitHub environment. A new Isaac Sim rollout is still required to validate the physical/simulation effect of the anchor change and inspect the generated trajectory CSV.

## Experimental protocol

Use the same checkpoint, dataset, validation episodes, seed, Isaac scene, speed/acceleration configuration, and 10 DDIM inference steps as the previous run. Compare the baseline branch with this branch sequentially. Do not retrain the checkpoint between A/B runs.

Primary metrics:
- tool-to-target distance over time;
- measured joint position/velocity versus requested command;
- predicted action nodes versus executor position;
- `limited_steps_since_sample` and cumulative limiting count;
- capture/collect outcome and timeout.

This is a diagnostic/control-handoff experiment, not evidence that the policy has learned robust apple picking. It does not change the model architecture, training data, checkpoint, or DDIM step count.
