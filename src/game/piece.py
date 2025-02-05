from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

class PieceType(Enum):
    """棋子类型枚举"""
    NORMAL = auto()    # 普通棋子
    FIRST_PLAYER = auto()  # 先手标记

class PieceColor(Enum):
    """棋子颜色枚举"""
    NONE = (0, (200, 200, 200))  # 用于先手标记
    BLUE = (1, (100, 140, 255))
    YELLOW = (2, (255, 230, 150))
    RED = (3, (255, 130, 130))
    BLACK = (4, (100, 100, 100))
    WHITE = (5, (240, 240, 240))

    def __init__(self, id_num: int, rgb: Tuple[int, int, int]):
        self.id = id_num
        self.rgb = rgb

    @property
    def rgb_value(self) -> Tuple[int, int, int]:
        """获取RGB颜色值"""
        return self.rgb

@dataclass
class Piece:
    """棋子类"""
    def __init__(self, color: PieceColor, piece_type: PieceType = PieceType.NORMAL):
        self.color = color
        self.piece_type = piece_type # 棋子类型:先手棋子还是普通棋子
        self.position = None  # 当前位置
        self.is_new = False    # 是否是新放置的棋子
        
    @property
    def is_first(self) -> bool:
        """是否是先手标记"""
        return self.piece_type == PieceType.FIRST_PLAYER
        
    @property
    def rgb(self) -> Tuple[int, int, int]:
        """获取瓷砖的RGB颜色值"""
        return self.color.rgb_value
    
    def move_to(self, row: int, col: int):
        """
        移动瓷砖到指定位置
        
        Args:
            row (int): 目标行
            col (int): 目标列
        """
        self.position = (row, col)
    
    def remove_from_board(self):  # TODO: 需要再看看
        """从棋盘上移除瓷砖"""
        self.position = None
    
    @classmethod
    def create_first_player_marker(cls) -> 'Piece':
        """创建先手标记"""
        return cls(color=PieceColor.NONE, piece_type=PieceType.FIRST_PLAYER)
        
    def __str__(self) -> str:
        if self.is_first:
            return "⭐"  # 先手标记使用星号表示
        return {
            PieceColor.BLUE: "🔵",
            PieceColor.YELLOW: "🟡",
            PieceColor.RED: "🔴",
            PieceColor.BLACK: "⚫",
            PieceColor.WHITE: "⚪",
        }.get(self.color, "?")
    
    def __repr__(self) -> str:
        """返回瓷砖的详细字符串表示"""
        pos_str = f" at {self.position}" if self.position else ""
        return f"Piece({self.color.name}{', FIRST' if self.is_first else ''}{pos_str})" 