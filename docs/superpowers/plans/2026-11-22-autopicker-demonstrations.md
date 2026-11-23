# Autopicker 单臂示范采集与回放计划

**目标：** 在已有研究分支实现可审计的单臂专家录制和命令回放，为策略训练提供首批数据。

**范围：** 沿用用户已认可的研究方案，完成 M0；本轮不训练网络。研究入口独立于演示入口，不修改 USD。原几何链、苹果管理及接果行为复用现有实现。普通 Python 负责数据审计，Isaac Python 负责实际采集与回放。

## 文件与职责

- `research/autopicker_dp/control.py`：关节单位、限位、速度/加速度约束及时间节点重采样。
- `research/autopicker_dp/episode.py`：NPZ/JSON 原始轨迹、事件与元数据的保存、载入、审计。先保留无额外依赖的原始格式，Zarr 训练转换留给 M1。
- `research/autopicker_dp/environment.py`：单臂 reset、专家路径、互斥动作入口、根据实际位姿判定抓取与收果。
- `research/autopicker_dp/capture.py`：锁步 RGB-D、渲染内外参、时刻与运动状态校验。
- `research/autopicker_dp/run.py`：collect/replay/audit 命令，失败归档和运行汇总。
- `research/tests/`：普通 Python 单元检查和 Isaac 实际采集/回放验收。
- `.gitignore`：忽略数据集、模型与运行输出。

## 执行顺序

- [x] 先写并运行控制/记录测试，验证命令不能越界、错误时间和旧帧被拒绝、失败回合不冒充成功、文件可往返读取。
- [x] 实现纯 NumPy 控制与原始数据存储，运行同一批测试。
- [x] 实现单臂 reset 与实际几何事件判定；专家、回放只通过同一执行器写关节目标。
- [x] 增加单相机采集，验证渲染等待不推进物理状态、相机参数对应当前实际姿态、时间戳单调。
- [x] 用 Isaac Sim 执行一个完整录制/回放回合，修正实际接口问题后扩大到至少五目标、十回合。
- [x] 检查 3 mm/1° 接近判据、正确苹果收果、其余三臂静止、数据完整性及场景资产未修改。
- [x] 补充运行命令、实测结果和未完成部分；提交研究分支。

## 验证约定

普通检查使用 `python -m unittest discover -s research/tests -v`。Isaac 入口通过安装目录的 `python.sh` 或 `python.bat` 执行 `research/autopicker_dp/run.py`。仿真运行必须同时检查退出码、JSON 汇总与明确的 PASS 标记，不把进程正常退出当成收果成功。

回放只读动作轨迹，不能读取专家阶段来决定抓放，也不能调用 IK 修正轨迹。若低频动作未通过收果验收，记录失败，调整采样/执行方案后重采，不修改成功容差。
