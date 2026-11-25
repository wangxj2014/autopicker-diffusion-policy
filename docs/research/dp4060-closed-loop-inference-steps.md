# RTX 4060 闭环 DDIM 步数对照

本实验把 DDIM 推理步数从独立推理进程参数贯通到 Isaac rollout。它与 `benchmark_inference.py` 的纯前向基准分开：前者测模型调用耗时，后者记录真实闭环抓取/收果结果。

## 重要前提

- 必须使用与 `datasets/autopicker/lowdim_v3` manifest SHA256 匹配的 checkpoint。
- 上一步 `dp_v2_smoke_4060_normfix_v2` 只有 10 次训练更新，只能用于验证软件链路，不能用于判断抓取质量。先训练完整 pilot checkpoint。
- 数据集目前只有 10 个回合、5 个目标、单场景、R_down、低维 oracle state。val 两回合属于同一个留出目标的重复，不等于跨苹果/跨场景泛化。
- 训练、Isaac Sim 和 rollout 串行执行；RTX 4060 8 GB 不并行占用。
- `--seed-mode episode` 按试验 seed、来源回合和局部推理请求编号固定噪声，便于不同步数配置比较。每个配置单独启动一次 Isaac。

## 1. 完整训练一个可评估 checkpoint

先确认工作区干净并记录当前分支/提交。不要覆盖 smoke 目录。

~~~bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py \
  --dataset datasets/autopicker/lowdim_v3 \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp_v3_full_4060_steps_v1 \
  --split train \
  --steps 3000 \
  --batch-size 32 \
  --seed 0 \
  --device cuda \
  --learning-rate 1e-4
~~~

训练结束必须出现 `AUTOPICKER_TRAIN_AND_RELOAD_PASS`。保留完整 `run_manifest.json`、`metrics.jsonl` 和 checkpoint；确认 `git_dirty=false`。若训练意外失败，不要复用该输出目录，另建带后缀的目录。

## 2. 运行相同验证回合的三组闭环

先设置 Isaac Sim Python 路径（按本机安装位置调整）：

~~~bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh
~~~

依次运行 5、10、20 步，不要并发：

~~~bash
"$ISAAC_SIM_PYTHON" research/autopicker_dp/rollout.py \
  --mode policy --dataset datasets/autopicker/lowdim_v3 \
  --checkpoint runs/autopicker/dp_v3_full_4060_steps_v1/checkpoint.pt \
  --inference-python work/autopicker_train_env/bin/python \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp_v3_policy_val_5step_v1 \
  --split val --seed 0 --seed-mode episode --inference-steps 5 \
  --timeout 35 --headless

"$ISAAC_SIM_PYTHON" research/autopicker_dp/rollout.py \
  --mode policy --dataset datasets/autopicker/lowdim_v3 \
  --checkpoint runs/autopicker/dp_v3_full_4060_steps_v1/checkpoint.pt \
  --inference-python work/autopicker_train_env/bin/python \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp_v3_policy_val_10step_v1 \
  --split val --seed 0 --seed-mode episode --inference-steps 10 \
  --timeout 35 --headless

"$ISAAC_SIM_PYTHON" research/autopicker_dp/rollout.py \
  --mode policy --dataset datasets/autopicker/lowdim_v3 \
  --checkpoint runs/autopicker/dp_v3_full_4060_steps_v1/checkpoint.pt \
  --inference-python work/autopicker_train_env/bin/python \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp_v3_policy_val_20step_v1 \
  --split val --seed 0 --seed-mode episode --inference-steps 20 \
  --timeout 35 --headless
~~~

每次输出必须有 `AUTOPICKER_ROLLOUT_COMPLETE`，并检查 `summary.json` 的 `completed=true`、`attempted=requested`。抓取失败仍留在分母。策略运行不使用专家 IK；与 `hybrid` 的结果分开统计。

## 3. 导出三个小型结果包

~~~bash
work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/dp_v3_policy_val_5step_v1 \
  --destination docs/research/results/dp_v3_policy_val_5step_v1

work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/dp_v3_policy_val_10step_v1 \
  --destination docs/research/results/dp_v3_policy_val_10step_v1

work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/dp_v3_policy_val_20step_v1 \
  --destination docs/research/results/dp_v3_policy_val_20step_v1
~~~

审核导出包中的机器信息后再推送。分析时分别比较成功数、真实 capture/collect、超时、末端到苹果最近距离、姿态误差和策略调用 p50/p95/p99 延迟。只有在同一个 checkpoint、同一个验证回合、相同逐回合随机种子下比较，才作为这次步数消融结果。

## 当前结果的处理原则

最新纯前向测试中 5/10 步比 20 步快约 4×/2×，但相对 20 步输出的关节动作差异很大。20 步也只是数值参考，不是真值动作。因此，在新完整训练 checkpoint 的闭环数据出来之前，不把 5 或 10 步设为默认，也不将 20 步参考输出称为高质量策略。
