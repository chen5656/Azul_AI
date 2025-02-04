"""游戏配置"""

class GameConfig:
    # 游戏基础配置
    NUM_PLAYERS = 2
    NUM_DISKS = 5
    
    # 棋子配置
    PIECES_PER_COLOR = 20
    PIECES_PER_DISK = 4
    
    # 棋盘配置
    BOARD_SIZE = 5
    PENALTY_SLOTS = 7
    
    # 分值配置
    PENALTY_VALUES = [-1, -1, -2, -2, -2, -3, -3] 