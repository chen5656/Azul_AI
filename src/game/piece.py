from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

class PieceType(Enum):
    """瓷砖类型枚举"""
    NORMAL = auto()  # 普通瓷砖
    FIRST_PLAYER = auto()  # 先手标记

class PieceColor(Enum):
    """瓷砖颜色枚举"""
    BLUE = (100, 140, 255)
    YELLOW = (255, 230, 150)
    RED = (255, 130, 130)
    BLACK = (100, 100, 100)
    WHITE = (240, 240, 240)
    NONE = (200, 200, 200)  # 用于先手标记

    def __init__(self, r: int, g: int, b: int):
        self.rgb = (r, g, b)

    @property
    def rgb_value(self) -> Tuple[int, int, int]:
        """获取RGB颜色值"""
        return self.rgb

@dataclass
class Piece:
    """
    瓷砖类
    
    Attributes:
        color (PieceColor): 瓷砖颜色
        piece_type (PieceType): 瓷砖类型
        position (Optional[Tuple[int, int]]): 瓷砖在棋盘上的位置 (row, col)
    """
    
    color: PieceColor
    piece_type: PieceType = PieceType.NORMAL
    position: Optional[Tuple[int, int]] = None
    
    @property
    def is_first(self) -> bool:
        """是否为先手标记"""
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
    
    def remove_from_board(self):
        """从棋盘上移除瓷砖"""
        self.position = None
    
    @classmethod
    def create_first_player_marker(cls) -> 'Piece':
        """
        创建先手标记
        
        Returns:
            Piece: 先手标记瓷砖
        """
        return cls(
            color=PieceColor.NONE,
            piece_type=PieceType.FIRST_PLAYER
        )
    
    def __str__(self) -> str:
        """返回瓷砖的字符串表示"""
        if self.is_first:
            return "F"
        return self.color.name[0]
    
    def __repr__(self) -> str:
        """返回瓷砖的详细字符串表示"""
        pos_str = f" at {self.position}" if self.position else ""
        return f"Piece({self.color.name}{', FIRST' if self.is_first else ''}{pos_str})" 