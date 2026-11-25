# RTX 4060 DDIM 推理步数基准

## 目的

使用同一个训练 checkpoint、同一组验证观测、相同随机种子，在 RTX 4060 上比较 DDIM 5/10/20 步推理。该实验只衡量策略模型的 GPU/主机前向推理耗时，以及动作输出相对 20 步参考结果的差异。

它不代表采摘成功率，也不等于真实部署端到端延迟。当前测量不包括 inference worker IPC、Isaac Sim stepping、相机采集和 ROS 2 执行；这些仍需通过 rollout 测量。

## 运行前提

- 使用干净的 Git 工作区和记录在 manifest 的 commit。
- 使用与 checkpoint 完全一致的数据集 manifest 哈希。
- 训练和 Isaac Sim 不要同时占用 RTX 4060；本实验只运行推理基准。
- 当前数据仍是单场景、R_down、低维 oracle state。结果只用来决定算力与去噪步数，不用于宣称视觉泛化。

## 命令

先运行测试：

~~~bash
work/autopicker_train_env/bin/python -m unittest discover \
  -s research/tests -p 'test_inference_benchmark.py' -v
~~~

再运行完整测试：

~~~bash
AUTOPICKER_DP_UPSTREAM="$PWD/work/diffusion_policy_upstream" \
work/autopicker_train_env/bin/python -m unittest discover \
  -s research/tests -v
~~~

选择已有训练 checkpoint 与当前配套数据集：

~~~bash
work/autopicker_train_env/bin/python research/benchmark_inference.py \
  --dataset datasets/autopicker/lowdim_v3 \
  --checkpoint runs/autopicker/dp_v2_smoke_4060_normfix_v2/checkpoint.pt \
  --upstream work/diffusion_policy_upstream \
  --output runs/autopicker/dp4060_inference_benchmark_normfix_v2 \
  --split val \
  --device cuda \
  --steps 5 10 20 \
  --reference-steps 20 \
  --max-samples 16 \
  --warmup 3 \
  --repeats 2 \
  --seed 0
~~~

实际 checkpoint 路径应以本机保留的模型文件为准。该脚本会校验 checkpoint 中的数据集 SHA256 与所选数据集 manifest 一致；不一致时立即退出。

## 输出文件

- run_manifest.json：机器、GPU、驱动、Git、checkpoint 和 dataset 身份。
- metrics.jsonl：每个验证观测、每个去噪步数及每次重复的推理耗时。
- system_metrics.csv：运行期间的 GPU、VRAM、CPU、RAM、温度和功耗。
- report.json：每种步数的 mean/std/p50/p95/p99/max 延迟、峰值显存，以及动作与 20 步参考的差异。

## 如何解读

1. 先看 p50、p95、p99；不要只比较平均值。
2. 对比 action_deviation_vs_reference。低步数与 20 步差异越大，表示数值求解结果发生变化；并不自动意味着动作质量变差或变好。
3. 只有在 Isaac Sim rollout 中保持同一目标集、同一随机种子和同一控制条件，才可以进一步比较采摘效果。
4. 如果 5 步显著缩短延迟且动作偏差小，可以把 5 步加入闭环对照；若偏差大，则先测试 10 步。
5. 纯 policy、hybrid 和 converted replay 必须分开统计。

## 导出日志到 GitHub

~~~bash
work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/dp4060_inference_benchmark_normfix_v2 \
  --destination docs/research/results/dp4060_inference_benchmark_normfix_v2
~~~

检查 manifest 和报告中没有不希望公开的本机信息后再提交。

