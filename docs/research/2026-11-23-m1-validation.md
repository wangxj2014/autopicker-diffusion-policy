# M1：低维 Diffusion Policy 流程与首次闭环实验

日期：2026-11-23。分支：`research/diffusion-policy`。实验基线为 `4edcb5e`，运行时为未提交开发状态；每个数据产物和 checkpoint 均记录源码 SHA256。最终两组训练记录的全部 Python 源码散列与本次提交代码一致。原始 `outputs/` 场景、演示控制器和相机功能没有修改。

**数据转换、训练、checkpoint 重载与 Isaac 推理接口已通过审计；学习策略尚未通过采摘验收。** 不能把损失下降或接口跑通写成学习采摘成功。机器可读结果见 [JSON 报告](2026-11-23-m1-validation.json)。

## 实测结果

| 实验 | 训练或执行范围 | 抓取 / 回合 | 收果 / 回合 |
| --- | --- | ---: | ---: |
| 新专家采集 | 5 个目标，每个重复两次，收果后记录 0.2 s | 10/10 | 10/10 |
| 转换后动作回放 | 10 Hz 节点，经同一 480 Hz 执行器 | 10/10 | 10/10 |
| 全量拟合策略 | 10 个回合训练，5 个已见目标各评估一次 | 0/5 | 0/5 |
| train-only 策略验证 | 3 个训练目标；1 个留出目标重复评估两次 | 0/2 | 0/2 |

七个学习策略回合均完整执行到 35 s 仿真超时，未发生推理进程中断，也没有隐藏失败回合。全量拟合模型最近末端距离依次为 10.759、53.766、73.736、87.758、202.925 mm；验证模型两回合为 76.059 和 77.797 mm。成功标准保持 3 mm/1° 抓取，再进入接果区域；没有放宽阈值或调用专家 IK 补救。

全量拟合模型包含所有目标，因此其结果只属于拟合诊断。验证集两回合属于同一颗苹果，不是两个独立场景。test 组已做离线诊断，并参与 all-pilot 诊断模型训练；本轮未对 train-only 模型运行 test 闭环，也不能把这批数据称为论文的独立最终测试集。

## 数据、模型与复现条件

- Isaac Sim 5.1，RTX 4060 Laptop 8 GiB；Isaac Python 3.11 与训练 Python 3.10 分进程运行。
- 有效原始数据：`datasets/autopicker/pilot_lowdim_tail_v1`，10 回合、5 目标。此次低维采集不录图像；M0 的 1,316 帧 RGB-D 及同步验收数据仍保留。
- 有效训练数据：`datasets/autopicker/lowdim_v2`，1,326 条 10 Hz 样本；按 `(scene_sha256, target_path)` 排序分成 3/1/1 目标、6/2/2 回合。
- manifest SHA256：`f7d96a5015010969051af29532cbc0d2203b0cc852703fb0b5dec0a43c704fd5`。
- 输入为 15 维真值状态：q、qdot、基座系目标/接果位置、attached；不输入阶段、时间、目标 ID。两帧相隔 0.1 s，预测 16 步，执行索引 `[1:3]`，0.2 s 重规划。
- 使用[官方 ConditionalUnet1D 的固定版本](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，通道 64/128/256、扩散嵌入 64、global conditioning、100 步 cosine epsilon 训练、20 步 DDIM、EMA。属于项目适配，不是官方 benchmark 原样复现。
- 两组均为 seed=0、3,000 optimizer steps、batch=64、AdamW lr=1e-4，未按此次闭环结果继续调参或追加训练。

| 训练 | 初始 100 步平均 loss | 最后 100 步平均 loss | 训练墙钟 | checkpoint 重载误差 |
| --- | ---: | ---: | ---: | ---: |
| all-pilot | 0.752319 | 0.013293 | 41.00 s | 0 |
| train-only | 0.761963 | 0.011917 | 42.76 s | 0 |

训练时 PyTorch 峰值已分配张量显存约 118 MiB；这不是进程总显存或 Isaac 加模型的总占用。两组 checkpoint 分别位于 `runs/autopicker/m1_overfit_v2`、`runs/autopicker/m1_train_v2`，权重散列及全部依赖版本见 JSON。

teacher-forced 离线 MAE 使用专家观察，不代表闭环误差；角度与位移分开报告：

| 模型 / 划分 | roll (rad) | hor (m) | ver (m) | rot (rad) |
| --- | ---: | ---: | ---: | ---: |
| all-pilot / 全部 | 0.01439 | 0.00492 | 0.00619 | 0.03280 |
| train-only / train | 0.01450 | 0.00362 | 0.00598 | 0.03145 |
| train-only / val | 0.03360 | 0.07805 | 0.03046 | 0.11628 |
| train-only / test | 0.20183 | 0.12176 | 0.06219 | 0.25568 |

全量模型 875 次推理请求平均 68.7 ms、P95 72.7 ms；验证模型 350 次平均 65.6 ms、P95 70.6 ms、最大 153.3 ms。这里包含 JSON 往返与推理耗时，且仿真锁步等待；没有证明端到端实时运行能力。每 35 s 仿真回合实际耗时约 47–52 s。

## 本轮发现与修复

1. **末尾采样丢失收果命令。** 原始 M0 回合在 collect 后立即停止，若截去最后不足 0.1 s 的片段，转换回放为 0/10 收果。例如第一回合最终末端停在接果位置外约 32 mm。采集器现额外记录 0.2 s 真实保持状态，转换器拒绝截掉 collect 的数据；重新采集后的回放为 10/10。旧 `lowdim_v1`、`m1_*_v1` 训练及 `m1_converted_replay_v2` 属于诊断产物，不用于最终 M1 结论。
2. **跨进程 Python 环境污染。** Isaac 的 `PYTHONHOME/PYTHONPATH` 导致训练 Python 导入错误版本标准库。推理子进程使用 `-I` 隔离，并在 Isaac 启动环境下实测 JSON 请求/回复通过。
3. **动作块边界遗漏。** 重新规划时保留上一块最后一个请求节点作为插值起点，再由共同执行器限速，避免丢掉边界节点；有针对性回归测试。
4. **配置及重载一致性。** JSON 的元组/列表按序列语义比较；重载恢复 checkpoint 的 scheduler 配置。固定种子 EMA 输出逐位一致。

## 学习失败的证据与下一轮

存在两个明确需要处理的问题，尚不能据此认定全部失败只有一个原因。

**动作输出范围限制了留出目标。** 当前 train action min/max 归一化与 `clip_sample=True` 联用，使 DDIM 输出限制在训练动作包络内。训练 hor 范围为约 `[0.1200, 0.4115] m`；验证专家动作最多低于下界 39.3 mm，测试动作最多超出包络 45.9 mm。这是可直接验证的表示限制，不能解释成模型已具备外推能力。下一轮先比较使用已知物理关节限位的固定动作归一化，保持观测归一化仅拟合训练集，并重新检查物理限位。

**已见目标也存在闭环分布偏移。** 全量拟合模型在已见目标上同样未抓取，因此范围问题不是全部原因。第一回合的 hor 速度约在 `[-0.276, 0.481] m/s`，明显超出专家样本 `[-0.0648, 0.1921] m/s`；175 次推理观测中有 53 次 hor 速度超出专家范围。模型在专家状态下的动作误差，不能保证自己产生的状态下继续可靠控制。

建议下一轮按顺序执行：固定动作范围归一化消融；补充初态/接近误差与恢复示范；在已见目标上比较单步执行与双步执行及有限训练/去噪预算。每次变更单独保存配置和种子，先达到可复现的闭环采摘，再扩大独立目标和场景。继续使用实际 capture/collect 门槛，不以降低标准或 IK 兜底替代学习成功。

## 验证与产物

- 42 项研究自动化测试通过（含 Zarr、PyTorch、消息过期/超时、跨回合清理及失败计数）；历史 M0 数据验收再次通过。
- 新采集器另完成 1/1 回合 RGB-D 检查，记录 140 帧，包含 collect 后的真实帧；映射同步误差为 0，相机/物理状态审计通过。数据位于 `datasets/autopicker/m1_rgbd_tail_smoke`。
- 10 个转换回放和 7 个策略回合均保存完整物理状态，源码/数据/checkpoint 散列及实际结果经过 `validate_m1.py` 审计。
- 本轮主代理完成代码复查；独立代理复审因服务不可用未完成。没有把它记为已通过。
- 数据、权重、大日志留在本地忽略目录；Git 只保存实现、测试、操作说明和小型报告。

复查命令（从仓库根目录执行）：

```bash
AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream work/autopicker_train_env/bin/python -m unittest discover -s research/tests -v
work/autopicker_train_env/bin/python research/tests/validate_m1.py --dataset datasets/autopicker/lowdim_v2 --replay runs/autopicker/m1_converted_replay_v3 --overfit-training runs/autopicker/m1_overfit_v2 --train-training runs/autopicker/m1_train_v2 --overfit-rollout runs/autopicker/m1_policy_overfit_v2 --val-rollout runs/autopicker/m1_policy_val_v2 --rgbd-smoke datasets/autopicker/m1_rgbd_tail_smoke --output work/m1-recheck.json
```

`AUTOPICKER_M1_AUDIT_PASS` 表示产物和统计一致，**不表示学习策略通过采摘验收**。M1 的稳健闭环采摘目标仍保持未完成。
