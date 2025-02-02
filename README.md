# Azul_AI

这是一个基于人工智能的Azul棋盘游戏项目。通过结合深度神经网络、遗传算法和强化学习，训练AI代理来学习和掌握Azul游戏的策略。

## 游戏简介

Azul是一个抽象的策略性瓷砖放置游戏，玩家需要收集瓷砖并按照特定模式放置在自己的板上来获得分数。游戏结束时，得分最高的玩家获胜。

### 游戏特点
- 2-4名玩家
- 回合制策略游戏
- 需要规划和预测对手行动
- 包含多种得分方式

## 项目结构

    Azul_AI/
    ├── src/
    │   ├── game/           # 游戏核心实现
    │   │   ├── board.py    # 游戏板和瓷砖放置逻辑
    │   │   ├── player.py   # 玩家类实现
    │   │   └── game_state.py # 游戏状态管理
    │   ├── ai/             # AI相关实现
    │   │   ├── neural_network.py    # 神经网络模型
    │   │   ├── genetic_algorithm.py # 遗传算法
    │   │   └── agent.py    # AI代理
    │   └── utils/          # 工具函数
    │       └── visualizer.py # 游戏可视化
    ├── tests/              # 测试文件
    ├── data/               # 数据存储
    │   ├── models/         # 训练模型存储
    │   └── training_logs/  # 训练日志
    └── docs/               # 文档

## 技术方案

### 1. 深度神经网络 (DNN)
- 输入层：游戏状态矩阵（包括当前棋盘状态、可用瓷砖等）
- 隐藏层：多层卷积神经网络处理空间特征
- 输出层：所有可能动作的概率分布

### 2. 遗传算法
- 染色体编码：神经网络权重
- 适应度函数：游戏胜率和得分
- 选择机制：锦标赛选择
- 交叉和变异：自适应率

### 3. 强化学习策略
- 状态空间：当前游戏局面
- 动作空间：所有合法移动
- 奖励机制：
  - 即时奖励：每步得分
  - 延迟奖励：最终游戏结果
  - 惩罚：非法移动

## 环境配置

### 依赖要求
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- Pygame 2.1+ (用于可视化)

### 安装步骤

1. 克隆仓库：

    git clone https://github.com/your-username/Azul_AI.git
    cd Azul_AI

2. 使用Conda创建环境：

    conda env create -f environment.yml
    conda activate azul-ai

3. 验证安装：

    python -c "import torch; import numpy; import pygame"

## 开发路线图

### Phase 1: 游戏环境适配 (进行中)
- [x] 项目结构搭建
- [x] 创建基础文档
- [ ] 实现游戏核心逻辑
- [ ] 实现状态向量化
- [ ] 实现动作空间定义
- [ ] 实现奖励系统
- [ ] 添加环境测试用例

### Phase 2: AI代理框架 (计划中)
- [ ] 实现基础Agent类
- [ ] 实现神经网络模型
- [ ] 实现经验回放缓冲区
- [ ] 实现遗传算法框架

### Phase 3-5: 待进行

## 当前进度
- 完成项目基础架构设计
- 正在实现游戏核心逻辑
- 准备开始编写基础测试用例

## 参与贡献
欢迎贡献代码、报告问题或提出建议。请遵循以下步骤：
1. Fork 项目
2. 创建特性分支 (git checkout -b feature/AmazingFeature)
3. 提交更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目维护者：[Huajun Chen]
- 项目链接：[https://github.com/your-username/Azul_AI](https://github.com/your-username/Azul_AI)
