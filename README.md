# PenguinFishBattle

企鹅、海豹与鱼：基于 PettingZoo Parallel API 的多智能体强化学习项目。三个企鹅智能体在 10 × 10 网格中捕食鱼并躲避海豹，使用独立 Q 学习（IQL / DQN）训练策略。

## 环境与安装

当前开发环境为 Python 3.14。依赖版本记录在 `requirements.txt`，图形界面使用 pygame-ce（导入名称为 `pygame`）。

```powershell
git clone https://github.com/shangruiwu/PenguinFishBattle.git
cd PenguinFishBattle
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch 默认自动选择可用的 CUDA 设备，否则使用 CPU。使用 GPU 时请按 PyTorch 官方说明安装与本机驱动匹配的版本。

## 运行

请在项目根目录执行，模型和输出图片的路径相对于当前工作目录。

```powershell
# 随机策略可视化
python test_random.py

# 对比随机策略、固定地图训练模型、随机地图训练模型
python evaluate_v11.py

# 固定地图模型评估与可视化
python test_iql.py

# 训练随机地图模型（默认 1500 回合）
python train_iql.py
```

`train_iql.py` 中的 `RANDOM_INIT` 控制随机初始布局；`MODEL_DIR` 和 `PLOT_FILE` 控制保存位置。默认训练会覆盖 `models_random/` 中的模型和对应训练图；需要保留已有结果时，请先修改输出路径。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| `penguin_env.py` | 网格世界、企鹅智能体、海豹、鱼、奖励与渲染 |
| `train_iql.py` | 各智能体独立的 DQN、经验回放与训练 |
| `test_random.py` | 随机策略演示 |
| `test_iql.py` | 固定地图模型评估与可视化 |
| `evaluate_v11.py` | 四组策略 / 地图组合评估，每组默认 300 回合 |
| `models/` | 固定地图训练模型，三个 `.pt` 文件 |
| `models_random/` | 随机地图训练模型，三个 `.pt` 文件 |
| `iql_*.png` | 已有训练结果图 |

## 环境接口

每个企鹅可执行停留、上、下、左、右五种动作。每个智能体的观测为 14 维坐标向量，包含三个企鹅、一个海豹和三条鱼的位置；缺失对象使用 `-1` 表示。环境支持固定与随机初始布局，每回合最多 200 步，渲染模式包括 `human` 与 `rgb_array`。

本仓库保留原有代码、训练模型与结果图，不包含虚拟环境、IDE 设置和 Python 缓存。

## 项目路线图

### V1 — 环境构建与基础训练

**目前进行中。**

目标：建立可靠的多智能体环境，完成 Independent DQN，并验证它在随机地图下的泛化能力。

| 版本 | 内容 | 学习知识 | 状态 |
| --- | --- | --- | --- |
| V1.0 | 三企鹅、鱼、海豹、Pygame 可视化 | ParallelEnv、Observation、Action、Reward、Termination | — |
| V1.1 | Independent DQN 与训练统计 | Independent Q-Learning、Replay Buffer、Target Network、Non-stationarity | — |
| V1.2 | 随机初始化地图，重新训练 IQL | Generalization、Exploration、过拟合 | — |
| V1.3 | 固定模型与随机模型的完整测试 | Benchmark、Evaluation、各 Agent 表现差异 | — |
| V1.4 | 改善环境和 Reward 设计 | Reward Shaping、Reward Hacking、Credit Assignment 入门 | 待开始 |
| V1.5 | 标准化实验和保存训练指标 | 多随机种子、方差、训练稳定性、环境测试 | 待开始 |

V1.0–V1.3 的状态暂未单独标注。

**V1.3 要完成四组实验：**

1. Random baseline。
2. 固定模型测试固定地图。
3. 固定模型测试随机地图。
4. 随机模型测试随机地图。

**V1.4 重点研究距离奖励是否真的有帮助。** 例如势函数奖励：

$$
F(s,s')=\gamma\Phi(s')-\Phi(s)
$$

可以选择鱼的距离构造 $\Phi$，但需要考虑鱼被其他企鹅吃掉、海豹风险、状态终止时的处理，避免设计出容易被反复刷分的奖励。

**V1 完成标准：** 三只企鹅在随机地图下表现优于随机策略，能够稳定复现实验结果，同时我们知道各自的学习差异来自哪里。

### V2 — Competitive MARL：真正开始算法对比

目标：保留三只企鹅争抢有限资源的竞争规则，研究不同学习算法的表现。

| 版本 | 内容 | 学习知识 |
| --- | --- | --- |
| V2.0 | 建立算法比较框架 | 相同环境、训练预算和测试种子 |
| V2.1 | IQL 升级为 Independent Double DQN | Q-value Overestimation、Double Q-Learning |
| V2.2 | 使用 Independent PPO 训练三只企鹅 | Actor-Critic、On-policy vs Off-policy |
| V2.3 | 尝试 Parameter Sharing PPO | 参数共享、Agent ID、经验共享 |
| V2.4 | 分析三只企鹅的竞争行为 | Non-stationarity、Best Response、策略分化 |

建议加入一个实验：**固定两只企鹅的策略，只训练第三只企鹅。** 再和三只企鹅同时训练的结果比较，帮助我们更清楚地理解非平稳性。

V2 完成后，应真正掌握 IQL 和 IPPO 在竞争任务上的区别，而不仅是能够运行代码。

### V3 — Cooperative MARL：合作学习与 CTDE

这是整个项目最重要的算法升级阶段。

目标：修改游戏规则，让三只企鹅一起争取团队收益。例如，总目标可以是尽可能多地吃鱼，并尽量减少被抓数量。

| 版本 | 内容 | 学习知识 |
| --- | --- | --- |
| V3.0 | 把环境改成 Team Reward | Cooperative MARL、Global Reward |
| V3.1 | 实现 VDN | Value Decomposition、团队 Q 值分解 |
| V3.2 | 实现 QMIX | Mixing Network、Monotonicity、CTDE |
| V3.3 | 实现 MAPPO | Centralized Critic、Decentralized Actors |
| V3.4 | 比较 IQL、VDN、QMIX、MAPPO | Credit Assignment、Sample Efficiency、Cooperation |

**VDN：**

$$
Q_{\mathrm{tot}}=\sum_{i=1}^{3}Q_i
$$

**QMIX：**

$$
Q_{\mathrm{tot}}=f_{\mathrm{mix}}(Q_1,Q_2,Q_3,s)
$$

并满足：

$$
\frac{\partial Q_{\mathrm{tot}}}{\partial Q_i}\geq 0
$$

在这个阶段，将 QMIX、MAPPO 的数学原理与实际代码中的网络结构对应起来。

**V3 完成标准：** 在明确的团队奖励下，比较至少三种多智能体学习算法，并研究是否出现有意义的合作行为。

### V4 — Partial Observability：部分可观测

目前三只企鹅能够直接看到整个地图。从 V4 开始，企鹅只能看见周围一部分区域。

| 当前：全局观测 | V4：局部观测 |
| --- | --- |
| 直接看到整个地图 | 只能看到自身可视范围内的区域 |

| 版本 | 内容 | 学习知识 |
| --- | --- | --- |
| V4.0 | 限制企鹅的可视范围 | Partial Observability、Dec-POMDP |
| V4.1 | 引入历史观测与记忆 | Recurrent PPO、GRU/LSTM |
| V4.2 | 企鹅之间发送有限消息 | Multi-Agent Communication |
| V4.3 | 比较不同观测半径和通信方式 | Information Sharing、Communication Cost |

研究问题：企鹅看不到所有鱼时，如何通过记忆和合作找到资源？

### V5 — Multi-Agent Adversarial RL：让海豹也学习

前面的海豹一直是规则驱动的：`70% 追最近企鹅 + 30% 随机`。这个阶段让海豹也变成一个 RL Agent，环境从 3 个 RL Agent 升级为 4 个。

| 🐧 🐧 🐧 Penguin Team | 对抗 | 🦭 Seal Agent |
| --- | --- | --- |
| 吃鱼、存活、合作 | VS | 追逐、捕获、预测 |

| 版本 | 内容 | 学习知识 |
| --- | --- | --- |
| V5.0 | 海豹从规则驱动变成 RL Agent | Asymmetric Multi-Agent RL |
| V5.1 | 分别训练企鹅与海豹 | Adversarial Training |
| V5.2 | 使用 Self-Play / 历史策略池 | Self-Play、Opponent Sampling |
| V5.3 | 分析不同对手下的策略表现 | Robustness、Opponent Generalization |
| V5.4 | 研究合作与竞争同时存在的机制 | Mixed-Motive MARL |

这个阶段引入真正意义上的多智能体对抗：企鹅内部可以合作，企鹅与海豹之间竞争。

### V6 — Research Experiments：把项目做成科研原型

最后一个大版本围绕一个明确的问题设计实验。

| 版本 | 内容 | 学习知识 |
| --- | --- | --- |
| V6.0 | 确定 Research Question 与 Hypothesis | 研究问题、基线设计 |
