# RTX 4060 smoke-test analysis and normalization follow-up

## Evidence inspected

Source bundle: `docs/research/results/dp_v2_smoke_4060`

The uploaded run used:
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB reported VRAM
- Ubuntu 22.04.5, Python 3.10.19
- PyTorch 2.7.0, CUDA runtime 12.6
- 10 optimization steps, batch size 4, learning rate 1e-4, seed 0
- Dataset: 10 episodes, 1326 rows, 0.1 s sampling interval, arm R_down
- Target-group split: 3 train target IDs, 1 validation target ID, 1 test target ID
- Training fit: 6 episodes / 820 rows / 820 sequence windows

The run completed checkpoint serialization and reload with zero observed prediction difference in its saved probe. The final training loss reported is 1.0852768 after only 10 updates; this is a pipeline smoke test, not evidence of convergence or policy quality.

## Important limits in the uploaded evidence

- The run manifest records `git.dirty=true`; therefore the run is not a clean-checkout reproduction.
- The training took about 0.517 s, while the telemetry interval was 1 s. Only one system sample was captured, so it cannot characterize peak utilization, VRAM, temperature, or power.
- The uploaded bundle contains a training report and metrics, but no captured unittest output and no policy rollout report. This upload does not establish that the complete unit-test suite passed or that the learned policy picked apples.
- The dataset is a low-dimensional oracle-state dataset (`q`, `qdot`, oracle goal/receiver positions in base coordinates, and oracle attachment flag). It is not yet an RGB-D perception policy. All examples are from R_down and one scene; multiple episodes repeat the same five target identities.

## Normalizer issue found

Observation channel order is:
- 0:4 joint position `q`
- 4:8 joint velocity `qdot`
- 8:11 oracle goal position in base frame
- 11:14 receiver position in base frame
- 14 attachment flag

For channel index 12 (the Y component of `receiver_base_m`), the fitted training range was approximately:
- min: `-0.3813296258 m`
- max: `-0.3813295066 m`
- spread: `1.1920929e-7 m`

The old exact-only constant check treated this float32 representation jitter as a variable feature and assigned a scale of `16,777,216`. That magnifies quantization noise. The follow-up branch treats a dimension as near-constant when its spread is no greater than:

```
8 * float32_epsilon * max(1, abs(train_min), abs(train_max))
```

Near-constant dimensions use scale 1 and bias `-train_min`. Inputs are still not clipped, so a meaningful held-out displacement remains represented in physical units. A regression test covers the observed numeric case and verifies the transform remains invertible.

## Telemetry follow-up

The resource monitor now takes a final sample when stopped. This makes sub-second smoke tests produce a start and end sample even when the periodic interval is one second. It is still a sparse endpoint measurement, not a high-resolution peak monitor. Formal runs lasting several minutes should use the periodic samples and PyTorch peak-memory fields together.

## Next experiment protocol

1. Run the unit tests on the RTX 4060 machine from a clean branch/commit.
2. Rebuild or reload the training normalizer with the updated constant-dimension logic.
3. Run a modest training baseline (for example 1000 updates) with no observation noise, keeping Isaac Sim closed.
4. Run validation and test rollouts using the saved checkpoint, with mode and split recorded explicitly.
5. Only then compare qdot noise ablations or architecture changes. Keep pure-policy and hybrid/IK-assisted results separate.
6. Expand target identity, tree/scene, viewpoint, and arm coverage before drawing conclusions about orchard generalization.

No claim of improved success rate is made by the normalizer change; it is a numerical-stability correction motivated by the uploaded report.
