# Autopicker 单臂示范、Diffusion Policy 训练与评估

这是研究入口。它使用开发版场景和现有几何模型，提供单臂专家、命令回放、低维 Diffusion Policy 训练和真实仿真闭环评估。普通演示仍通过 `outputs/autopicker_start.py` 启动。

M0 已通过十回合录制与十回合同目标回放，覆盖五个目标；实测指标及限制见[验收报告](../docs/research/2026-11-23-m0-validation.md)。

## 环境

采集与实际回放需要 Isaac Sim 5.1 自带 Python。数据审计及单元测试只需要普通 Python 和 NumPy。命令从仓库根目录执行，以下 `/path/to/isaac-sim/python.sh` 应替换为安装路径；Windows 使用 `C:\isaacsim\python.bat`。

## 采集和回放

先采集 R_down 单臂、五个可达目标的十个回合。每个回合都重置场景和关节，同一目标重复两次用于检验复现性，不代表十个独立场景。

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/run.py collect --output datasets/autopicker/pilot_rgbd_v2 --episodes 10 --targets 5 --speed-scale 3 --timeout 35 --headless
```

使用 10 Hz 位置节点回放，指令由同一限位/限速/限加速度执行器插值到 480 Hz。回放根据真实末端位置和方向触发抓取与收果，不读取录制时的阶段或抓放事件来伪造成功。

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/run.py replay --input datasets/autopicker/pilot_rgbd_v2 --output runs/autopicker/replay_v2 --replay-dt 0.1 --no-rgbd --timeout 35 --headless
```

回放命令中的 `--no-rgbd` 表示不重复生成图像；仍执行完整 PhysX 运动与收果判断。去掉它即可同时录制回放的图像。`--sample-dt` 控制图像采样间隔，默认 0.1 s，须为 1/480 s 的整数倍；`--replay-dt` 独立控制回放节点间隔。

输出目录必须不存在，程序不会覆盖旧数据。`--episodes 1 --targets 1` 可做短检查。`--arm` 可选择其他机械臂，但首轮验收范围是 R_down。默认轨迹速度为 1，示例使用 3；驱动增益保持当前演示 `speed_scale=45` 对应的增益，避免低速轨迹伴随增益降低产生静态下垂。完整参数写入每回合元数据。

```bash
python research/autopicker_dp/run.py audit --input datasets/autopicker/pilot_rgbd_v2
python research/tests/validate_pilot.py datasets/autopicker/pilot_rgbd_v2 runs/autopicker/replay_v2
python -m unittest discover -s research/tests -v
```

验收需要 `AUTOPICKER_COLLECT_PASS`、`AUTOPICKER_REPLAY_PASS` 和 `AUTOPICKER_M0_ACCEPTANCE_PASS`，同时检查 `summary.json`。部分 Isaac 启动/关闭路径可能返回 0，即使 Python 抛出了异常，不能仅检查退出码。

## 数据内容

每个 `episode_0000/` 包含：

- `episode.json`：场景和源码 SHA256、关节单位、初态、整园重置检查、配置、坐标变换、抓取/收果事件、成功或失败原因及数据文件校验值。
- `states.npz`：480 Hz 实测关节位置/速度、实际下发的位置/速度命令、末端与苹果/接果位置、全部十六关节和附着状态。最后一行是终止状态，不再执行它后面的动作。
- `rgbd.npz`：160×120 RGB、米制光轴深度、有效掩码、内外参、原始渲染时间、映射后的仿真时间及对应物理步。

成功必须包含同一颗苹果的 capture 和 collect；抓取误差不超过 3 mm，方向误差不超过 1°。其余三臂保持初始关节位置，漂移单独记录。失败回合保留原因；若连数据审计条件都不满足，则保存 `failure.json`，不生成可用于训练的成功样本。

每回合分配目标前，检查全部苹果恢复初始世界/局部位置、可见且可用，落果副本隐藏且关闭碰撞；任意苹果位置偏差超过 1 μm 就停止。重置时先解除上一回合的苹果绑定，避免初始化机械臂时误移旧目标。早期本地 `pilot_rgbd_v1` 为排查数据，不用于最终验收。

## 时间与控制约定

Isaac 的 Writer `reference_time` 在这里属于渲染时钟，实测不等于 `World.current_time`。采集在固定物理状态下完成同步渲染，验证等待前后物理时间与全部关节均不变，并将渲染相机矩阵与 PhysX 相机刚体位姿比较。保留原始时间，另存 `clock_mapping=lockstep_render_barrier`、`exposure_sim_time_s` 和对应物理步。映射误差为零表示同一次锁步事务，没有证明异步运行或真实硬件的时间同步精度。

采集与回放均通过同一个执行器写关节位置和速度目标。研究进程不启动普通演示控制器的物理订阅。专家使用既有 FK/IK 生成几何轨迹；回放仅消费保存的命令，重置过程中的接果位置初始化仍复用现有模型。

## M1：准备低维训练数据

M1 使用苹果/接果位置和附着状态的真值，尚未使用图像学习。先在收果后额外记录 0.2 s 的真实物理状态，确保取整到 10 Hz 时保留最后的放果命令。`--post-collect-hold` 默认 0.2，只用于研究采集；普通演示不增加停留。原来的 `pilot_rgbd_v2` 仍用于 M0 验收，但末尾记录不足，不能直接作为当前训练数据。

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/run.py collect --output datasets/autopicker/pilot_lowdim_tail_v1 --episodes 10 --targets 5 --speed-scale 3 --post-collect-hold .2 --timeout 35 --no-rgbd --headless
```

此轮用 `--no-rgbd` 加快低维接口验证；视觉策略采集时去掉它。不要改写历史输出目录。

训练使用独立 Python 3.10 环境，Isaac 继续使用自带 Python 3.11。已验证 PyTorch 2.7.0+cu126、NumPy 2.2.6；其余固定版本见 `requirements-training.txt`。示例新环境安装方式：

```bash
python3.10 -m venv work/autopicker_train_env
work/autopicker_train_env/bin/python -m pip install -r research/requirements-training.txt
git clone https://github.com/real-stanford/diffusion_policy.git work/diffusion_policy_upstream
git -C work/diffusion_policy_upstream checkout --detach 5ba07ac6661db573af695b419a7947ecb704690f
work/autopicker_train_env/bin/python research/autopicker_dp/dataset.py --input datasets/autopicker/pilot_lowdim_tail_v1 --output datasets/autopicker/lowdim_v2
```

本机复用了已有 `mj310` 环境中的 CUDA PyTorch，通过 `venv --system-site-packages` 建立训练环境，仅把补充依赖安装到 venv。命令启动时直接调用该 venv 的 Python，无需切换 Isaac 环境。

转换器审计原始回合、重置状态、控制配置及文件 SHA256。导出 `data.zarr`、`manifest.json`、`normalizer.json` 和完整校验清单。五个目标按路径排序分成 3/1/1 个目标组，对应 6/2/2 回合；同一目标的重复回合不会跨组。这只是小样本接口试验，不能用来证明泛化。

- 输入 15 维：关节位置 4、速度 4、基座坐标系下的目标位置 3、接果位置 3、附着状态 1。坐标使用 USD 行向量逆变换；不输入阶段、时间或目标编号。
- 在 `t` 观察，标签是 `t+0.1 s` 实际下发的绝对关节位置命令。只保留完整间隔，末尾状态另存元数据；若取整丢掉 collect 命令则拒绝转换。
- 两个观测间隔 0.1 s，预测长度 16，每次执行预测索引 `[1:3]` 两个动作节点，0.2 s 重新规划。边界复制带有效掩码，窗口不跨回合。
- 常规训练的归一化仅拟合 train；不裁剪观测。动作经扩散模型输出、反归一化和限位检查后，由原有执行器在 480 Hz 下执行。

## 训练、checkpoint 和离线诊断

```bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py --dataset datasets/autopicker/lowdim_v2 --upstream work/diffusion_policy_upstream --output runs/autopicker/m1_train_v2 --split train --steps 3000 --batch-size 64 --seed 0
work/autopicker_train_env/bin/python research/autopicker_dp/evaluate.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m1_train_v2/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m1_train_v2/offline.json
```

代码直接导入官方仓库的 `ConditionalUnet1D` 并核对 commit 与源码状态。此处采用较小网络（通道 64/128/256，扩散嵌入 64）、全局观测条件、100 步 cosine epsilon 训练、20 步 DDIM 推理和 EMA；这是针对本项目的适配，不是官方基准的原样复现。

checkpoint 保存模型、EMA、优化器、配置、scheduler、归一化、随机种子/torch RNG、数据和源码散列、依赖版本。训练结束后重新加载 EMA，并核对固定种子输出完全相同；须看到 `AUTOPICKER_TRAIN_AND_RELOAD_PASS`。当前没有断点续训 CLI。

可用不同输出目录加 `--split all-pilot` 做全十回合拟合诊断，例如 `runs/autopicker/m1_overfit_v2`。该 checkpoint 使用所有目标，相关结果不能称为留出测试。离线评估使用专家观测，报告每个关节各自单位下的 MAE/RMSE，不把米和弧度混成一个指标，也不把它当采摘成功率。

## Isaac 闭环评估

先确认转换后的动作节点可以完成收果：

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode converted-replay --dataset datasets/autopicker/lowdim_v2 --output runs/autopicker/m1_converted_replay_v3 --split all-pilot --timeout 35 --headless
```

再运行训练策略：

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m1_train_v2/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m1_policy_val_v2 --split val --seed 0 --timeout 35 --headless
```

`--split` 选择初始目标/状态；`--max-episodes` 可限制评估数量。推理在独立 Python 进程中运行，以隔离 Isaac 的 Python 环境。仿真等待推理时不推进时间；每次消息检查回合编号、时间和请求编号。每回合保存完整物理状态、推理观测/动作/延迟和真实 capture/collect 事件；策略动作不调用专家 IK。重置时仍复用原模型初始化。

必须同时检查 `AUTOPICKER_ROLLOUT_COMPLETE` 和 `summary.json` 的 `completed/requested/attempted/successes`。COMPLETE 表示评估跑完，**不代表全部成功**；超时、非法动作或数据审计失败都保留在分母中。成功仍需满足 3 mm/1° 抓取阈值并到达接果装置，没有放宽阈值。

```bash
AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream work/autopicker_train_env/bin/python -m unittest discover -s research/tests -v
```

普通 Python 可运行数据/控制接口测试；Zarr、PyTorch 测试会在缺依赖时跳过，完整检查应使用上述训练环境。实测结果和下一步问题见 [M1 实验报告](../docs/research/2026-11-23-m1-validation.md)。

## M2：动作范围对照与恢复示范

`--action-normalization data` 保留 M1 的示范 min/max（也是默认值）；`physical` 使用数据清单中的实际关节上下限。两者观测归一化都只拟合所选训练回合。物理范围版本在 checkpoint 中保存独立动作变换，不改写原始数据集的 normalizer。保持 DDIM clipping 时，物理范围允许模型输出示范包络外的合法关节命令，但不保证精度或闭环稳定性。

等预算训练与离线诊断：

```bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py --dataset datasets/autopicker/lowdim_v2 --upstream work/diffusion_policy_upstream --output runs/autopicker/m2_train_physical_v1 --split train --action-normalization physical --steps 3000 --batch-size 64 --seed 0
work/autopicker_train_env/bin/python research/autopicker_dp/evaluate.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m2_train_physical_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m2_train_physical_v1/offline.json
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m2_train_physical_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m2_policy_val_physical_v1 --split val --seed 0 --timeout 35 --headless
```

全量拟合对照将训练改为 `--split all-pilot`，输出为 `m2_overfit_physical_v1`；闭环使用该 checkpoint、`--split all-pilot --max-episodes 5`，输出为 `m2_policy_overfit_physical_v1`。其余参数相同。结果见 [M2 实验报告](../docs/research/2026-11-23-m2-validation.md)。

恢复采集仅选择原始 **train** 的三个目标，每个目标取首次成功示范中 capture 前 0.5 s 的实测关节状态，分别施加正负 `[3°, 25 mm, 20 mm, 5°]` 偏移。初始速度归零后物理稳定，再由专家完成接近、采摘和放果。越限扰动被拒绝并保存失败记录，后续回合继续；不会偷偷裁剪或加入留出目标。

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/recovery.py --dataset datasets/autopicker/lowdim_v2 --output datasets/autopicker/m2_recovery_raw_v1 --timeout 35 --headless
work/autopicker_train_env/bin/python research/autopicker_dp/dataset.py --input datasets/autopicker/m2_recovery_raw_v1 --output datasets/autopicker/m2_recovery_lowdim_v1 --split-counts 3 0 0
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode converted-replay --dataset datasets/autopicker/m2_recovery_lowdim_v1 --output runs/autopicker/m2_recovery_replay_v1 --split train --timeout 35 --headless
work/autopicker_train_env/bin/python research/tests/validate_m2.py --include-recovery --output work/m2-recheck.json
```

只有全部采集成功才执行此处的转换和回放；失败回合保留在原始目录中，转换器会拒绝包含失败的目录。`AUTOPICKER_RECOVERY_COMPLETE` 表示所有请求已处理，不表示全部成功。静态扰动恢复数据没有注入运动中扰动，也不是 DAgger；本轮归一化对照不混入这批新数据，避免同时改变两个实验因素。

## M3：恢复示范训练消融

只增加 M2 的六条静态恢复示范，主数据集仍为 `lowdim_v2`。整个观测归一化及物理动作映射固定为原 train 拟合结果；新窗口不重新拟合 normalizer。训练从原 820 个窗口与恢复 685 个窗口的合集中均匀抽样，共 1,505 个窗口，恢复窗口占 45.51%。这不是 50/50 混合；固定总更新数也意味着原始示范的平均采样次数减少。

```bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py --dataset datasets/autopicker/lowdim_v2 --recovery-dataset datasets/autopicker/m2_recovery_lowdim_v1 --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_train_recovery_v1 --split train --action-normalization physical --steps 3000 --batch-size 64 --seed 0
work/autopicker_train_env/bin/python research/autopicker_dp/evaluate.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m3_train_recovery_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_train_recovery_v1/offline.json
```

`--recovery-dataset` 仅允许配合 `--split train --action-normalization physical`。接入时从原始文件重建并核对导出数组，检查目标划分、源回合/数据散列、扰动状态、实际稳定初态和重复轨迹；checkpoint 同时保存主数据集与恢复集身份及采样规则。恢复集的原始来源也必须保留。

按顺序执行三组闭环实验，每组都重新启动 Isaac；不要与 GPU 训练或其他推理任务并行：

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m2_train_physical_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_baseline_train_v1 --split train --max-episodes 3 --seed 0 --timeout 35 --headless
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m3_train_recovery_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_recovery_train_v1 --split train --max-episodes 3 --seed 0 --timeout 35 --headless
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m3_train_recovery_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_recovery_val_v1 --split val --seed 0 --timeout 35 --headless
work/autopicker_train_env/bin/python research/tests/validate_m3.py --output work/m3-recheck.json
```

前两组配对使用原 train 索引 `[1,2,4]`，即三个不同已见目标；第三组验证索引 `[0,5]` 为同一留出目标重复两次，与 M2 已保存的 `m2_policy_val_physical_v1` 对比。这轮新增 8 次闭环评估，复用 2 次 M2 验证结果；没有新增 test 闭环。`AUTOPICKER_M3_AUDIT_PASS` 只表示产物与实验条件一致。结论见 [M3 实验报告](../docs/research/2026-11-23-m3-validation.md)。

## M4：保留运动状态的专家纠偏

`dynamic_recovery.py` 从 M3 三个 train 策略失败轨迹的固定 2 s、6 s 时刻提取全关节位置、活动臂实测速度和上一物理步执行器位置/速度命令。先重置果园，再恢复这些量，恢复后不推进物理或再次稳定。旧日志没有其他三臂的速度，明确置零；这不是完整 PhysX 求解器快照，也不是在线 DAgger。

按顺序采集、导出和回放，仅在六条采集与六条回放全部成功后训练：

```bash
/path/to/isaac-sim/python.sh research/autopicker_dp/dynamic_recovery.py --dataset datasets/autopicker/lowdim_v2 --policy-rollout runs/autopicker/m3_recovery_train_v1 --checkpoint runs/autopicker/m3_train_recovery_v1/checkpoint.pt --output datasets/autopicker/m4_dynamic_raw_v1 --timeout 35 --headless
work/autopicker_train_env/bin/python research/autopicker_dp/dataset.py --input datasets/autopicker/m4_dynamic_raw_v1 --output datasets/autopicker/m4_dynamic_lowdim_v1 --split-counts 3 0 0
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode converted-replay --dataset datasets/autopicker/m4_dynamic_lowdim_v1 --output runs/autopicker/m4_dynamic_replay_v1 --split train --timeout 35 --headless
work/autopicker_train_env/bin/python research/autopicker_dp/train.py --dataset datasets/autopicker/lowdim_v2 --recovery-dataset datasets/autopicker/m4_dynamic_lowdim_v1 --upstream work/diffusion_policy_upstream --output runs/autopicker/m4_train_dynamic_v1 --split train --action-normalization physical --steps 3000 --batch-size 64 --seed 0
work/autopicker_train_env/bin/python research/autopicker_dp/evaluate.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m4_train_dynamic_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m4_train_dynamic_v1/offline.json
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m4_train_dynamic_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m4_dynamic_train_v1 --split train --max-episodes 3 --seed 0 --timeout 35 --headless
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m4_train_dynamic_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m4_dynamic_val_v1 --split val --seed 0 --timeout 35 --headless
work/autopicker_train_env/bin/python research/tests/validate_m4.py --output work/m4-recheck.json
```

动态示范替换 M3 的静态示范，两份恢复集不叠加。训练仍冻结原 train 的完整 normalizer；原 820 个窗口加动态 733 个窗口，共 1,553 个，均匀采样时动态占 47.20%。目标划分、网络、seed=0、3000 步和成功门槛沿用 M3。新增五次策略试验与 M3 已保存的三次 train、两次 val 配对比较，没有新增 test 闭环。数据来源的策略轨迹、父示范和 checkpoint 都必须保留用于散列核验；输出目录必须为新目录。实际结果见 [M4 实验报告](../docs/research/2026-11-23-m4-validation.md)。

## M5：直接预测干净动作

`--prediction-type sample` 将扩散网络的训练目标改为干净的归一化动作序列（x0），DDIM 同时使用 `sample` 解码。默认仍为 `epsilon`；旧 checkpoint 不含此配置时按 epsilon 加载，模型与保存的 scheduler 目标不一致时拒绝推理。两种目标的 loss 数值不可直接比较，须比较物理单位误差与完整采摘结果。

同预算对照使用 M3 的原 820 + 静态恢复 685 个窗口、完整冻结 normalizer、seed=0、3000 步和相同网络/执行器。固定观测诊断抽取原 train 的每八个窗口，使用噪声种子 0–7，分别统计采样波动、平均预测偏差与单次采样 MAE；它只测专家观测下的输出，不能代替闭环成功率。

```bash
work/autopicker_train_env/bin/python research/autopicker_dp/train.py --dataset datasets/autopicker/lowdim_v2 --recovery-dataset datasets/autopicker/m2_recovery_lowdim_v1 --upstream work/diffusion_policy_upstream --output runs/autopicker/m5_train_sample_v1 --split train --action-normalization physical --prediction-type sample --steps 3000 --batch-size 64 --seed 0
work/autopicker_train_env/bin/python research/autopicker_dp/evaluate.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m5_train_sample_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m5_train_sample_v1/offline.json
work/autopicker_train_env/bin/python research/autopicker_dp/diagnose.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m5_train_sample_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m5_train_sample_v1/diagnostic.json
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m5_train_sample_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m5_sample_train_v1 --split train --max-episodes 3 --seed 0 --timeout 35 --headless
/path/to/isaac-sim/python.sh research/autopicker_dp/rollout.py --mode policy --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m5_train_sample_v1/checkpoint.pt --inference-python work/autopicker_train_env/bin/python --upstream work/diffusion_policy_upstream --output runs/autopicker/m5_sample_val_v1 --split val --seed 0 --timeout 35 --headless
work/autopicker_train_env/bin/python research/autopicker_dp/diagnose.py --dataset datasets/autopicker/lowdim_v2 --checkpoint runs/autopicker/m3_train_recovery_v1/checkpoint.pt --upstream work/diffusion_policy_upstream --output runs/autopicker/m3_train_recovery_v1/m5_diagnostic.json
work/autopicker_train_env/bin/python research/tests/validate_m5.py --output work/m5-recheck.json
```

所有 GPU 命令串行执行。3000 步模型未完成采摘，因此额外执行一次预定的 30000 步实验：训练从相同 seed 重新开始，训练输出改为 `m5_train_sample_long_v1`、步数改为 `30000`；离线评估及诊断使用该 checkpoint，闭环输出分别为 `m5_sample_long_train_v1`、`m5_sample_long_val_v1`。其余参数完全相同；审计加 `--include-longer`。没有启用 CUDA 严格确定性，不保证两次训练的前 3000 步逐位一致。这组增加了训练预算，单独报告，不能算作同预算目标消融。审计检查 checkpoint、目标、训练预算、数据来源、探针协议、两次 sample 训练的源码一致性和全部尝试，`AUDIT_PASS` 本身不表示成功率提升。完整结果见 [M5 实验报告](../docs/research/2026-11-23-m5-validation.md)。

## 范围与下一步

这些是固定场景、单臂、真值输入和视觉吸附条件下的工程验证。即使拟合成功，也仍需要更多独立目标/场景、扰动示范、多个训练种子、基线及视觉感知实验，才能支持论文结论。

数据集、日志、模型和训练输出被 Git 忽略，GitHub 只保存代码、说明与小型验收报告。复制数据时应保留 `episode.json` 及全部 NPZ 文件。
