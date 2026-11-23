# Autopicker 单臂示范采集与回放

这是研究入口。它使用开发版场景和现有几何模型，运行单臂专家或预录命令回放；普通演示仍通过 `outputs/autopicker_start.py` 启动。这里尚无 Diffusion Policy 网络或训练命令。

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

## 范围与下一步

这些是使用苹果真值和视觉吸附的确定性专家示范，属于接口验收数据，不是论文中的学习成功率或泛化结果。原始 NPZ 也尚未转换为研究草案中的 10 Hz Zarr 训练样本；下一阶段需要按目标/场景划分数据、明确动作对齐、适配 Dataset，再训练和评估低维 Diffusion Policy。

数据集、日志、模型和训练输出被 Git 忽略，GitHub 只保存代码、说明与小型验收报告。复制数据时应保留 `episode.json` 及全部 NPZ 文件。
