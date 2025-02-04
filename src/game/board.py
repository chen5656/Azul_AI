from typing import List, Optional
from .piece import Piece, PieceColor
from .config import GameConfig
import warnings
from functools import wraps

def deprecated_params(*params):
    """
    标记函数参数为已弃用的装饰器
    
    Args:
        params: 要标记为已弃用的参数名称
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for param in params:
                if param in kwargs:
                    warnings.warn(
                        f"Parameter '{param}' is deprecated and will be removed in future versions.",
                        DeprecationWarning,
                        stacklevel=2
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class PlayerBoard:
    """
    玩家棋盘类
    
    Attributes:
        name (str): 玩家名称
        score (int): 玩家得分
        prep_area (List[List[Optional[Piece]]]): 准备区,5行,每行长度从1到5
        scoring_area (List[List[Optional[Piece]]]): 得分区,5x5的网格
        penalty_area (List[Optional[Piece]]): 扣分区,最多容纳7个瓷砖
    """
    
    # 结算区颜色模式（5x5网格）
    SCORING_PATTERN = [
        [PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE],
        [PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK],
        [PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED],
        [PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW],
        [PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE]
    ]
    
    # 扣分区扣分规则
    PENALTY_POINTS = [-1, -1, -2, -2, -2, -3, -3]
    
    @deprecated_params('x', 'y')
    def __init__(self, x: int, y: int, name: str, config: GameConfig = None):
        """
        初始化玩家棋盘
        
        Args:
            name (str): 玩家名称
            config (GameConfig): 游戏配置
            
        Deprecated Args:
            x (int): [已弃用] 棋盘在屏幕上的x坐标, 用于可视化
            y (int): [已弃用] 棋盘在屏幕上的y坐标, 用于可视化
        """
        self.config = config or GameConfig()
        self.name = name
        self._position = (x, y)  # 标记为内部使用
        
        # 初始化各个区域
        self.prep_area = [[None] * (i + 1) for i in range(5)]  # 准备区（三角形）:每行i+1个位置
        self.scoring_area = [[None] * 5 for _ in range(5)]  # 结算区 5 x 5
        self.penalty_area = []  # 扣分区:可以放超过7个棋子,但只有前7个计算分数
        self.score = 0
        
    def can_place_pieces(self, row: int, pieces: List[Piece]) -> bool:
        """
        检查是否可以在准备区的指定行放置棋子。
        要求至少有一个空位,多余的棋子会进入扣分区。
        
        Args:
            row (int): 准备区的行索引 (0-4)
            pieces (List[Piece]): 要放置的棋子列表
        
        Returns:
            bool: 如果可以放置返回True
        """
        if not pieces:
            return False            
            
        # 获取非先手标记的棋子
        normal_pieces = [p for p in pieces if not p.is_first]
        if not normal_pieces:
            return True  # 只有先手标记时可以放置  -- TODO:是否需要修改？
            
        # 检查颜色一致性
        color = normal_pieces[0].color
        if not all(p.color == color for p in normal_pieces):
            return False
        
        # 检查是否整排为空
        if all(p is None for p in self.prep_area[row]):            
            # 检查结算区该行是否已有相同颜色
            if any(p is not None and p.color == color for p in self.scoring_area[row]):
                return False
            else:
                return True
        
        # 检查最后一格是否为空
        if self.prep_area[row][-1] is not None:
            return False
                    
        # 检查第一格颜色是否一致
        if self.prep_area[row][0].color != color:
            return False
            
        return True
        
    def add_pieces_to_prep_area(self, row: int, pieces: List[Piece]) -> List[Piece]:
        """
        填充准备区,返回多余的棋子。
        从最左边的空位开始连续填充棋子。
        
        Args:
            row (int): 准备区的行索引 (0-4)
            pieces (List[Piece]): 要放置的棋子列表
            
        Returns:
            List[Piece]: 无法放置的多余棋子
        """
        # 找到最左边的空位
        row_size = len(self.prep_area[row])
        first_empty = 0
        while first_empty < row_size and self.prep_area[row][first_empty] is not None:
            first_empty += 1
            
        # 如果没有空位,返回所有棋子
        if first_empty == row_size:
            return pieces
            
        # 计算从first_empty到行尾的空位数量
        empty_count = row_size - first_empty
                
        # 计算可以放置的棋子数量
        pieces_to_place = pieces[:empty_count]
        remaining_pieces = pieces[empty_count:]
        
        # 从最左边的空位开始连续填充
        for i, piece in enumerate(pieces_to_place):
            self.prep_area[row][first_empty + i] = piece
            
        return remaining_pieces
        
    def add_to_penalty(self, pieces: List[Piece]) -> None:
        """
        将棋子添加到扣分区。可以放超过7个棋子,但只有前7个计算扣分。
        扣分将在最终计算时进行,而不是在添加时。
        
        Args:
            pieces (List[Piece]): 要添加的棋子列表
        """
        for piece in pieces:
            self.penalty_area.append(piece)
            
    def clear_penalty_area(self) -> List[Piece]:
        """
        清空扣分区并返回所有棋子
        
        Returns:
            List[Piece]: 扣分区中的所有棋子
        """
        pieces = self.penalty_area[:]
        self.penalty_area = []
        return pieces
        
    def score_row_round_end(self, row: int) -> int:
        """
        计算指定行的回合结束得分，只计算新放置棋子的得分
        
        规则:
        1. 单个瓷砖:1分
        2. 相邻连接:每个相邻+1分
        
        Args:
            row (int): 行索引
            
        Returns:
            int: 得分
        """
        if not 0 <= row < 5:
            return 0
            
        # 获取行中的瓷砖
        row_pieces = self.scoring_area[row]
        if not any(p is not None and p.is_new for p in row_pieces):  # 没有新棋子
            return 0
            
        score = 0
        
        # 只对新棋子计算分数
        for col, piece in enumerate(row_pieces):
            if piece is not None and piece.is_new:
                # 基础分
                score += 1
                # 与左侧相连
                if col > 0 and row_pieces[col-1] is not None:
                    score += 1
                # 与上方相连
                if row > 0 and self.scoring_area[row-1][col] is not None:
                    score += 1
                # 与下方相连
                if row < 4 and self.scoring_area[row+1][col] is not None:
                    score += 1
                    
        return score
        
    def score_row_game_end(self, row: int) -> int:
        """
        计算指定行的游戏结束额外得分
        
        规则:
        1. 完整行:额外2分
        
        Args:
            row (int): 行索引
            
        Returns:
            int: 得分
        """
        if not 0 <= row < 5:
            return 0
            
        # 检查是否是完整行
        if all(piece is not None for piece in self.scoring_area[row]):
            return 2
        return 0
        
    def score_column_round_end(self, col: int) -> int:
        """
        计算指定列的回合结束得分，只计算新放置棋子的得分
        
        规则:
        1. 单个瓷砖:1分
        2. 相邻连接:每个相邻+1分
        
        Args:
            col (int): 列索引
            
        Returns:
            int: 得分
        """
        if not 0 <= col < 5:
            return 0
            
        # 获取列中的瓷砖
        col_pieces = [self.scoring_area[row][col] for row in range(5)]
        if not any(p is not None and p.is_new for p in col_pieces):  # 没有新棋子
            return 0
            
        score = 0
        
        # 只对新棋子计算分数
        for row, piece in enumerate(col_pieces):
            if piece is not None and piece.is_new:
                # 基础分
                score += 1
                # 与上方相连
                if row > 0 and col_pieces[row-1] is not None:
                    score += 1
                # 与左侧相连
                if col > 0 and self.scoring_area[row][col-1] is not None:
                    score += 1
                # 与右侧相连
                if col < 4 and self.scoring_area[row][col+1] is not None:
                    score += 1
                    
        return score
        
    def score_column_game_end(self, col: int) -> int:
        """
        计算指定列的游戏结束额外得分
        
        规则:
        1. 完整列:额外7分
        
        Args:
            col (int): 列索引
            
        Returns:
            int: 得分
        """
        if not 0 <= col < 5:
            return 0
            
        # 检查是否是完整列
        if all(self.scoring_area[row][col] is not None for row in range(5)):
            return 7
        return 0
        
    def calculate_round_end_score(self) -> int:
        """
        计算回合结束时的得分,并更新玩家总分(self.score)
        
        Returns:
            int: 本回合得分（不包括之前累积的分数）
        """
        score = 0
        
        # 计算行得分
        for row in range(5):
            score += self.score_row_round_end(row)
            
        # 计算列得分
        for col in range(5):
            score += self.score_column_round_end(col)
            
        # 计算扣分
        score += self.calculate_penalty()
        
        # 更新玩家总分
        self.score += score
        
        # 重置所有棋子的is_new状态
        self._reset_new_pieces()
        
        return score
        
    def _reset_new_pieces(self):
        """
        重置所有棋子的is_new状态为False
        在每回合结算分数后调用
        """
        for row in range(5):
            for col in range(5):
                piece = self.scoring_area[row][col]
                if piece is not None:
                    piece.is_new = False
        
    def calculate_game_end_score(self) -> int:
        """
        计算游戏结束时的额外得分
        
        Returns:
            int: 游戏结束额外得分
        """
        score = 0
        
        # 计算完整行奖励
        for row in range(5):
            score += self.score_row_game_end(row)
            
        # 计算完整列奖励
        for col in range(5):
            score += self.score_column_game_end(col)
            
        # 计算颜色完成奖励
        for color in PieceColor:
            if color != PieceColor.NONE:
                score += self.calculate_color_bonus(color)
                
        return score
        
    def calculate_total_score(self) -> int:
        """
        计算总得分
        
        Returns:
            int: 总得分
        """
        self.score += self.calculate_round_end_score()
        return self.score
        
    def calculate_color_bonus(self, color: PieceColor) -> int:
        """
        计算指定颜色的完成奖励
        
        规则:完成一个颜色的所有瓷砖可以获得10分奖励
        
        Args:
            color (PieceColor): 瓷砖颜色
            
        Returns:
            int: 奖励分数
        """
        # 检查每行中指定颜色的位置是否都已放置瓷砖
        for row in range(5):
            col = self.SCORING_PATTERN[row].index(color)
            if self.scoring_area[row][col] is None:
                return 0
        return 10
        
    def calculate_penalty(self) -> int:
        """
        计算扣分区的扣分。只计算前7个棋子的扣分。
        
        Returns:
            int: 扣分值（负数）
        """
        penalty = 0
        # 只计算前7个棋子的扣分
        for i in range(min(len(self.penalty_area), self.config.PENALTY_SLOTS)):
            penalty += self.config.PENALTY_VALUES[i]
        return penalty 