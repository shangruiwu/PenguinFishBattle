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