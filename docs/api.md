# Azul AI API文档

## 游戏核心 (game)

### environment.py

#### AzulEnv
游戏环境类，遵循OpenAI Gym接口标准。

主要方法说明：
- `reset()`: 重置环境到初始状态
- `step()`: 执行一个动作并返回结果
- `render()`: 渲染当前游戏状态

### game_state.py

#### GameState
游戏状态枚举类。

状态说明：
- INIT: 游戏初始化状态
- RUNNING: 游戏进行中
- SCORING: 正在进行结算
- ROUND_END: 回合结束
- GAME_END: 游戏结束
- ERROR: 错误状态 