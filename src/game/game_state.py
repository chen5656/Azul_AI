from enum import Enum, auto

class GameState(Enum):
    """
    Azul游戏状态枚举类
    
    状态说明：
    - INIT: 游戏初始化状态
    - RUNNING: 游戏进行中
    - SCORING: 正在进行结算
    - ROUND_END: 回合结束
    - GAME_END: 游戏结束
    - ERROR: 错误状态
    """
    
    INIT = auto()        # 游戏初始化
    RUNNING = auto()     # 游戏进行中
    SCORING = auto()     # 结算阶段
    ROUND_END = auto()   # 回合结束
    GAME_END = auto()    # 游戏结束
    ERROR = auto()       # 错误状态
    
    def __str__(self):
        return self.name
    
    def is_terminal(self) -> bool:
        """
        检查当前是否为终止状态
        
        Returns:
            bool: 如果是GAME_END或ERROR则返回True
        """
        return self in [GameState.GAME_END, GameState.ERROR]
    
    def can_take_action(self) -> bool:
        """
        检查当前状态是否可以执行动作
        
        Returns:
            bool: 如果是RUNNING状态则返回True
        """
        return self == GameState.RUNNING
    
    def need_scoring(self) -> bool:
        """
        检查当前状态是否需要进行结算
        
        Returns:
            bool: 如果是SCORING状态则返回True
        """
        return self == GameState.SCORING 