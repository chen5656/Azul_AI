# Azul AI

使用深度强化学习训练AI来玩Azul棋盘游戏。

## 项目状态

- [x] 游戏环境实现完成
- [x] 游戏规则和记分系统实现完成
- [x] 基础测试用例编写完成
- [ ] AI训练框架开发中
- [ ] AI模型训练中

## 项目结构

```
src/
├── game/           # 游戏核心实现
│   ├── board.py    # 棋盘类
│   ├── piece.py    # 棋子类
│   ├── config.py   # 配置类
│   └── environment.py  # 游戏环境类
├── ai/             # AI相关实现
│   ├── models/     # 神经网络模型
│   ├── algorithms/ # 强化学习算法
│   ├── trainer.py  # 训练器
│   └── evaluator.py # 评估器
└── utils/          # 工具函数
tests/              # 测试用例
docs/               # 文档
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/azul-ai.git
cd azul-ai
```

2. 创建并激活虚拟环境：
```bash
conda env create -f environment.yml
conda activate azul
pip install -e .
```

## 使用方法

### 运行游戏环境

```python
from src.game.environment import AzulEnv

env = AzulEnv()
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # 随机动作，用于测试
    obs, reward, done, info = env.step(action)
    env.render()
```

### 训练AI代理（即将推出）

```python
from src.ai.trainer import Trainer
from src.ai.models import AzulNet
from src.ai.algorithms import PPO

# 创建模型和训练器
model = AzulNet()
trainer = Trainer(model, algorithm=PPO())

# 开始训练
trainer.train(num_episodes=10000)
```

## 开发路线图

1. 游戏环境实现 ✅
2. AI训练框架开发 🚧
3. 模型训练和优化 📅
4. Web界面开发 📅
5. 在线训练和对战系统 📅

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- OpenAI Gym 框架
- PyTorch 深度学习框架
- Azul 棋盘游戏设计者 Michael Kiesling


接下来，我们需要：
实现强化学习算法（如PPO）
实现经验收集和回放机制
实现训练循环和评估系统
你想从哪个部分开始？
