from typing import List, Optional, Tuple, Dict
from .piece import Piece, PieceColor

class PlayerBoard:
    """
    玩家棋盘类
    
    Attributes:
        x (int): 棋盘在屏幕上的x坐标
        y (int): 棋盘在屏幕上的y坐标
        player_name (str): 玩家名称
        score (int): 玩家得分
        prep_area (List[List[Optional[Piece]]]): 准备区，5行，每行长度从1到5
        scoring_area (List[List[Optional[Piece]]]): 得分区，5x5的网格
        penalty_area (List[Optional[Piece]]): 扣分区，最多容纳7个瓷砖
    """
    
    # 每行对应的颜色模式
    COLOR_PATTERN = [
        [PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE],
        [PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK],
        [PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW, PieceColor.RED],
        [PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE, PieceColor.YELLOW],
        [PieceColor.YELLOW, PieceColor.RED, PieceColor.BLACK, PieceColor.WHITE, PieceColor.BLUE]
    ]
    
    # 扣分区扣分规则
    PENALTY_POINTS = [-1, -1, -2, -2, -2, -3, -3]
    
    def __init__(self, x: int, y: int, player_name: str):
        """
        初始化玩家棋盘
        
        Args:
            x (int): 棋盘在屏幕上的x坐标
            y (int): 棋盘在屏幕上的y坐标
            player_name (str): 玩家名称
        """
        self.x = x
        self.y = y
        self.player_name = player_name
        self.score = 0
        
        # 初始化准备区 (每行长度不同: 1-5)
        self.prep_area = [
            [None] * (i + 1) for i in range(5)
        ]
        
        # 初始化得分区 (5x5)
        self.scoring_area = [
            [None] * 5 for _ in range(5)
        ]
        
        # 初始化扣分区
        self.penalty_area = [None] * 7
        
    def can_place_pieces(self, row: int, color: PieceColor) -> bool:
        """
        检查是否可以在指定行放置指定颜色的瓷砖
        
        Args:
            row (int): 目标行索引
            color (PieceColor): 瓷砖颜色
            
        Returns:
            bool: 如果可以放置则返回True
        """
        # 检查行索引是否有效
        if not 0 <= row < 5:
            return False
            
        # 检查该行是否已满
        if all(piece is not None for piece in self.prep_area[row]):
            return False
            
        # 检查该行是否已有其他颜色的瓷砖
        existing_pieces = [p for p in self.prep_area[row] if p is not None]
        if existing_pieces and existing_pieces[0].color != color:
            return False
            
        # 检查对应的得分区位置是否已被占用
        target_col = self.COLOR_PATTERN[row].index(color)
        if self.scoring_area[row][target_col] is not None:
            return False
            
        return True
        
    def add_pieces(self, row: int, pieces: List[Piece]) -> List[Piece]:
        """
        在准备区指定行添加瓷砖
        
        Args:
            row (int): 目标行索引
            pieces (List[Piece]): 要添加的瓷砖列表
            
        Returns:
            List[Piece]: 无法放置的瓷砖列表
        """
        if not self.can_place_pieces(row, pieces[0].color):
            return pieces
            
        # 计算可以放置的数量
        empty_spots = sum(1 for p in self.prep_area[row] if p is None)
        can_place = min(empty_spots, len(pieces))
        
        # 放置瓷砖
        placed = 0
        for i, spot in enumerate(self.prep_area[row]):
            if spot is None and placed < can_place:
                self.prep_area[row][i] = pieces[placed]
                pieces[placed].move_to(row, i)
                placed += 1
                
        # 返回剩余无法放置的瓷砖
        return pieces[placed:]
        
    def add_to_penalty(self, pieces: List[Piece]) -> List[Piece]:
        """
        将瓷砖添加到扣分区
        
        Args:
            pieces (List[Piece]): 要添加的瓷砖列表
            
        Returns:
            List[Piece]: 无法放置的瓷砖列表
        """
        # 找到第一个空位
        for i, spot in enumerate(self.penalty_area):
            if spot is None and pieces:
                self.penalty_area[i] = pieces[0]
                pieces[0].move_to(-1, i)  # 使用-1表示扣分区
                pieces = pieces[1:]
                
        return pieces
        
    def score_row(self, row: int) -> int:
        """
        计算指定行的得分
        
        规则：
        1. 单个瓷砖：1分
        2. 相邻连接：每个相邻+1分
        3. 完整行：额外2分
        
        Args:
            row (int): 行索引
            
        Returns:
            int: 得分
        """
        if not 0 <= row < 5:
            return 0
            
        # 获取行中的瓷砖
        row_pieces = self.scoring_area[row]
        if not any(row_pieces):  # 空行
            return 0
            
        score = 0
        consecutive_count = 0  # 连续瓷砖计数
        
        # 计算基础分和连续分
        for col, piece in enumerate(row_pieces):
            if piece is not None:
                consecutive_count += 1
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
                    
        # 完整行奖励
        if consecutive_count == 5:
            score += 2
            
        return score
        
    def score_column(self, col: int) -> int:
        """
        计算指定列的得分
        
        规则：
        1. 单个瓷砖：1分
        2. 相邻连接：每个相邻+1分
        3. 完整列：额外7分
        
        Args:
            col (int): 列索引
            
        Returns:
            int: 得分
        """
        if not 0 <= col < 5:
            return 0
            
        # 获取列中的瓷砖
        col_pieces = [self.scoring_area[row][col] for row in range(5)]
        if not any(col_pieces):  # 空列
            return 0
            
        score = 0
        consecutive_count = 0  # 连续瓷砖计数
        
        # 计算基础分和连续分
        for row, piece in enumerate(col_pieces):
            if piece is not None:
                consecutive_count += 1
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
                    
        # 完整列奖励
        if consecutive_count == 5:
            score += 7
            
        return score
        
    def calculate_color_bonus(self, color: PieceColor) -> int:
        """
        计算指定颜色的完成奖励
        
        规则：完成一个颜色的所有瓷砖可以获得10分奖励
        
        Args:
            color (PieceColor): 瓷砖颜色
            
        Returns:
            int: 奖励分数
        """
        # 检查每行中指定颜色的位置是否都已放置瓷砖
        for row in range(5):
            col = self.COLOR_PATTERN[row].index(color)
            if self.scoring_area[row][col] is None:
                return 0
        return 10
        
    def calculate_total_score(self) -> int:
        """
        计算总得分
        
        Returns:
            int: 总得分
        """
        score = 0
        
        # 计算行得分
        for row in range(5):
            score += self.score_row(row)
            
        # 计算列得分
        for col in range(5):
            score += self.score_column(col)
            
        # 计算颜色完成奖励
        for color in PieceColor:
            if color != PieceColor.NONE:
                score += self.calculate_color_bonus(color)
                
        # 计算扣分
        score += self.calculate_penalty()
        
        return score
        
    def calculate_penalty(self) -> int:
        """
        计算扣分区的扣分
        
        Returns:
            int: 扣分值（负数）
        """
        penalty = 0
        for i, piece in enumerate(self.penalty_area):
            if piece is not None:
                penalty += self.PENALTY_POINTS[i]
        return penalty 