# RTX 4060 单机 Diffusion Policy 实验日志

本方案用于 Isaac Sim、Diffusion Policy 训练和推理都运行在同一台 Ubuntu 电脑的情况。默认目标 GPU 是 RTX 4060；每次运行都会记录机器实际检测到的 GPU 型号、显存和驱动版本，不依赖预设值。

## 1. 单机 GPU 运行约束

第一轮对比实验请让训练和 Isaac Sim 闭环评估串行运行：

1. 训练前关闭 Isaac Sim，以及其他不必要的 CUDA 程序。
2. 运行训练，结束后确认生成 checkpoint 和 report。
3. 训练进程退出后再启动 Isaac Sim 闭环评估。
4. 不要在第一次基线实验中同时训练和运行 Isaac Sim。

原因是两者共享同一块 GPU 和显存。同时运行会改变训练吞吐、仿真帧率和推理延迟，容易把资源竞争误判成策略本身的变化。确认串行基线后，可以单独设计并发压力测试；并发测试应标记为另一种运行条件。

## 2. 自动生成的日志

训练 research/autopicker_dp/train.py 和闭环 research/autopicker_dp/rollout.py 都会在各自的输出目录自动创建以下文件：

| 文件 | 内容 |
| --- | --- |
| run_manifest.json | Git commit/branch、工作区是否有未提交修改、Ubuntu/Python、CPU/RAM、GPU/驱动、命令参数、输入文件 SHA256 |
| system_metrics.csv | 每秒系统 CPU、当前进程树 CPU/RSS、系统 RAM、GPU 利用率、显存占用、温度、功耗 |
| resource_monitor_errors.jsonl | nvidia-smi 或系统采样失败信息；没有错误时不会生成 |
| metrics.jsonl（训练） | 每 100 步 loss、梯度范数、学习率、当前 CUDA allocated/reserved 显存和累计耗时 |
| report.json（训练） | checkpoint、训练参数、数据哈希、峰值显存、资源监控汇总、checkpoint 重载一致性 |
| inference.log（闭环） | 独立推理 Python 进程的标准错误和初始化信息 |
| episode_*/episode.json | 每回合事件、状态审计、推理请求、观测、动作、推理延迟及采摘结果 |
| summary.json（闭环） | 试验模式、数据和 checkpoint 哈希、请求/成功数量、逐回合结果及资源监控汇总 |

system_metrics.csv 的 GPU 利用率和显存使用量是整张 GPU 的设备级指标；gpu_processes_json 每约 5 秒采集一次 NVIDIA 可见的计算进程 PID、名称和显存。进程树 CPU/RSS 覆盖当前运行进程及其子进程，因此 Isaac Sim 运行时会包括独立的 inference worker。显卡显示进程是否出现在进程列表中取决于驱动/NVIDIA-SMI 的可见性；设备级 GPU 利用率和总显存仍会保留。

监控采用 Python 标准库和 nvidia-smi，不额外安装 psutil。采样间隔为 1 秒，不进入仿真物理步或 EtherCAT 实时控制回路。

## 3. RTX 4060 训练建议

第一次测试继续使用小 batch：

- smoke test：--batch-size 4 --steps 10
- 正式训练起点：先尝试 --batch-size 8 或 16，确认峰值显存后再增加。
- 遇到 CUDA OOM 时，先降低 batch size，不要通过与 Isaac Sim 并行运行来测试。
- 训练日志同时记录 allocated/reserved 显存；system_metrics.csv 记录 GPU 全局显存和利用率，可以识别桌面/其他 CUDA 程序造成的显存竞争。

不在这里预设正式 batch size；最终值应以本机实际的峰值显存和训练吞吐为准。

## 4. 闭环评估重点

每个 episode 的 episode.json 应重点检查：

- inference_requests：观测、动作、每次端到端推理延迟。
- 事件序列：approach、capture、collect、failure 及对应仿真时间。
- success / reason：是否完成采摘和明确失败原因。
- wall_time_s：仿真运行在真实电脑上消耗的时间。
- summary.json：train/val/test、policy/hybrid 等试验必须分开比较。

纯 policy 与使用传统 IK 接管的 hybrid / hybrid-corridor 不能合并成功率。

## 5. 整理日志并提交 GitHub

数据集、runs/ 和模型权重保持在本地，不直接提交进普通 Git。每次实验完成后，从仓库根目录运行：

~~~bash
work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/你的实验目录 \
  --destination docs/research/results/你的实验目录
~~~

例如：

~~~bash
work/autopicker_train_env/bin/python research/export_run_report.py \
  --run runs/autopicker/m7_hybrid_val_s0 \
  --destination docs/research/results/m7_hybrid_val_s0
~~~

导出脚本只复制 run_manifest.json、summary.json、训练指标、资源 CSV、推理/进程日志，以及每回合的 JSON 审计；不会复制 checkpoint、数据集、RGB-D 帧、视频或轨迹 NPZ。单个文件超过 20 MiB 时会跳过，并在 bundle_manifest.json 中记录原因。

检查导出目录后提交：

~~~bash
git add docs/research/results/你的实验目录
git commit -m "Add RTX 4060 experiment diagnostics"
git push
~~~

导出目录刻意放在被跟踪的 docs/research/results/ 下；原始 runs/ 仍保持 .gitignore 保护。上传前请检查 run_manifest.json 中记录的本机路径和日志内容，确认没有不希望公开的信息。

## 6. 发回给优化分析的最小材料

优先提交完整导出目录。最关键的文件是：

1. run_manifest.json：确认代码版本、输入数据及 RTX 4060 环境。
2. summary.json：确认这次试验究竟运行了哪些回合、成功与失败情况。
3. system_metrics.csv：区分 GPU 显存不足、显卡利用率不足、CPU/RAM 瓶颈。
4. episode_*/episode.json：定位动作错误、延迟尖峰和失败阶段。
5. inference.log 或 resource_monitor_errors.jsonl：检查推理初始化和采样错误。

只有在这些日志表明具体瓶颈后，才继续改网络结构、推理步数、动作块长度或观测增强，避免同时改太多变量。
