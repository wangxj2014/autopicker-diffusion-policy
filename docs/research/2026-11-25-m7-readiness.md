# M7：提前精抓接管与逐回合随机数隔离

**代码与 CPU 检查已完成；新增闭环实验为 0/20，尚不能声称采摘成功率提高。** 首次启动在加载模型时因 CUDA 驱动故障退出，实际采摘尝试为 0。内核在本轮实验之前已记录 NVIDIA Xid 154，恢复动作是 `Node Reboot Required`。完整状态见[状态 JSON](2026-11-25-m7-status.json)。

日期：2026-11-25；基线提交 `29df26c`；分支 `research/diffusion-policy`。本轮没有重新训练。M5 sample 30000 checkpoint SHA256 为 `d0d90421470e49ac5c3828e5486ade87da326907a33f994d725b5fc18d47f85a`。

## 改动

新增实验模式 `--mode hybrid-corridor`。沿用 M6 的 30 mm 精抓入口；在此之外，只在实测距离 ≤120 mm、位于接近轴前方、横向偏差 ≤20 mm、末端方向误差 ≤5° 时提前接管。接管后锁存精抓目标，不因时间到或短暂外移重新启动网络。真实 capture 当帧才回撤，保持实测 hor 对应的接果位置。最终 IK 姿态仍为向上倾斜 30°，不声称关节插值构成严格直线的 Cartesian 接近轨迹，也未新增碰撞规划。

抓取阈值仍为 3 mm / 1°，收果阈值 25 mm，超时 35 s。默认 `hybrid` 保留 M6 行为，展示 USD、相机和四臂演示没有改动。这仍是“扩散粗接近 + 真值 IK 精抓 + 传统回撤”的混合控制。

新增 `--seed-mode episode`，默认仍是旧 `global` 模式。每次扩散噪声取 SHA256(`autopicker-episode-v1:<trial seed>:<source index>:<local request index>`) 的前四字节大端整数。局部序号每回合从 0 开始，全局 request_id 继续递增以检查过期响应；服务端重算噪声种子并回显全部身份字段。这样前一个回合提前接管不会改变后续回合的噪声。请求日志保存这些字段供审计。

## 历史轨迹诊断，不是新闭环结果

在五次原始 M6 日志上复算 FK、原控制命令和新入口条件。下表的“新条件首次满足”是在**原轨迹**上检查的假设接管时刻，没有执行接管后的新轨迹，不能视作新的采摘时间。

| 目标 / 来源索引 | M6 实际接管 (s) | 新条件首次满足 (s) | 当时距离 (mm) | 横向偏差 (mm) | 方向误差 (°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Apple_035 / train 1 | 18.304 | 3.123 | 87.986 | 12.692 | 4.968 |
| Apple_026 / train 2 | 7.304 | 3.031 | 33.183 | 19.938 | 3.887 |
| Apple_034 / train 4 | 32.383 | 3.129 | 107.468 | 19.971 | 2.371 |
| Apple_036 / val 0 | 未进入 | 未满足 | — | — | — |
| Apple_036 / val 5 | 未进入 | 未满足 | — | — | — |

三个候选状态的终点 IK 都通过位置与姿态残差检查。验证目标最接近时仍有约 73–76 mm 横向偏差、15–17° 方向误差，本改动不解决这个偏差。这里的阈值选择参考了历史轨迹，是开发诊断，不是独立验证。历史 M1–M5 也已使用过离线 test 数据，不能宣称现有 test 从未被查看。

真实 `serve.py` 子进程在 CPU 加载同一 checkpoint 完成四次请求：同一 source/local seed、相同观测，global request_id 从 1 改为 37 后，输出动作最大差为 0；响应身份字段全部一致。CPU 随机生成器与 CUDA 不同，这只验证协议和 CPU 确定性。可复算结果见[诊断 JSON](2026-11-25-m7-readiness.json)。

## 验证与待运行实验

98 项自动化测试通过，包括提前接管的距离/侧向/方向/前后约束、原 30 mm fallback、锁存、事件回撤、局部种子、回显拒绝、伪造入口/种子/轨迹前缀、断点续跑的源码一致性。独立代码预审还比较了当前默认 HybridPolicy 与提交 `29df26c` 的 50 个场景、每场景 35 次调用：命令逐位一致，事件和网络调用一致。审计入口在缺失实验记录时拒绝生成验收 JSON，没有把准备检查当作仿真验收。

已固定的实验矩阵为：M6 `hybrid` / M7 `hybrid-corridor` × inference seeds `[0,1]` × train 来源 `[1,2,4]`、val 来源 `[0,5]`，共 20 次新尝试。模型、重置、阈值、速度、时限相同；全部使用 episode seed。原 M6 的 global seed 结果不作为这一轮的随机数配对基线。比较成功数、capture/collect 时间、包含失败的 35 s 截断时间、接管前原始状态/观测/动作前缀及接管后命令重建。

第一次启动失败记录完整保留在本地 `runs/autopicker/m7_hybrid_train_s0_startup_failure_20261125`，其 summary 为 attempted=0；它是基础设施失败，不是采摘失败。CUDA 预检查再次失败时不创建试验目录。新 runner 串行运行、只复用来源和源码一致的已完成组，拒绝覆盖部分完成记录，最后调用严格的 20 次产物审计器。独立复核后补充两项边界保护：只有进程正常退出、出现完成标记且没有 traceback 时才生成绑定原始日志与摘要散列的 completion 凭据；恢复与最终验收都强制校验。每回合保存全部推理尝试及全局编号起止，非法 action 或服务失败所消耗的编号也保留，避免误拒绝下一个正常回合。最终独立复核重跑 98 项测试，并实际验证了 PolicyClient → LivePolicy → HybridPolicy → failure.json → 下一回合的编号衔接，以及真实 CPU 子进程清理异常后的凭据拒绝。该审计器目前通过负例单元测试，尚没有 M7 闭环产物可验收。

GPU 恢复后，在仓库根目录运行：

```bash
# 查看固定矩阵，不启动仿真
work/autopicker_train_env/bin/python research/run_m7.py
# CUDA 预检查、20 次串行实验及产物审计
work/autopicker_train_env/bin/python research/run_m7.py --execute
```

CPU 检查可随时复算：

```bash
AUTOPICKER_DP_UPSTREAM=work/diffusion_policy_upstream work/autopicker_train_env/bin/python -m unittest discover -s research/tests -v
work/autopicker_train_env/bin/python research/diagnostics/m7_readiness.py --output work/m7-readiness-recheck.json
```

历史 M6 的完整验证器绑定当时控制/推理源码，需在 `29df26c` 上复现其完整报告；本轮没有改写历史实验。M7 仍为 experimental。GPU 恢复后先完成上述矩阵，再判断是否接受提前接管；若验证目标仍无法入区，下一项研究应针对粗接近的横向纠偏，而不是放宽真实抓取判据。
