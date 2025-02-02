import os
import pathlib

def create_directory_structure():
    # 定义项目结构
    structure = {
        'src': {
            'game': ['__init__.py', 'board.py', 'player.py', 'game_state.py'],
            'ai': ['__init__.py', 'neural_network.py', 'genetic_algorithm.py', 'agent.py'],
            'utils': ['__init__.py', 'visualizer.py']
        },
        'tests': {
            '': ['__init__.py', 'test_game.py', 'test_ai.py']
        },
        'data': {
            'models': [],
            'training_logs': []
        }
    }

    # 创建目录和文件
    for main_dir, sub_dirs in structure.items():
        for sub_dir, files in sub_dirs.items():
            # 创建完整路径
            full_path = os.path.join(main_dir, sub_dir) if sub_dir else main_dir
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            
            # 创建文件
            for file in files:
                file_path = os.path.join(full_path, file)
                pathlib.Path(file_path).touch()

    # 创建根目录文件
    root_files = ['README.md', 'requirements.txt']
    for file in root_files:
        pathlib.Path(file).touch()

    # 写入requirements.txt的内容
    requirements_content = """numpy>=1.19.2
tensorflow>=2.4.0
pygame>=2.0.1
matplotlib>=3.3.2
pytest>=6.2.4
"""
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)

    # 写入README.md的内容
    readme_content = """# Azul_AI

这是一个使用人工智能来玩Azul棋盘游戏的项目。该项目结合了神经网络和遗传算法来训练AI代理，使其能够学习和掌握Azul游戏的策略。

## 项目结构

- `src/game/`: 包含Azul游戏的核心实现
  - `board.py`: 游戏板和瓷砖放置的实现
  - `player.py`: 玩家类的实现
  - `game_state.py`: 游戏状态管理
- `src/ai/`: AI相关实现
  - `neural_network.py`: 神经网络模型实现
  - `genetic_algorithm.py`: 遗传算法实现
  - `agent.py`: AI代理实现
- `src/utils/`: 工具函数
  - `visualizer.py`: 游戏可视化工具

## 安装

1. 克隆仓库： """


if __name__ == '__main__':
    create_directory_structure()
    print("项目结构创建完成！")