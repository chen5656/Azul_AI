"""游戏渲染器"""

from typing import Dict, List, Optional, Tuple
import pygame
from .board import PlayerBoard
from .piece import Piece, PieceColor
from .environment import AzulEnv

class AzulRenderer:
    """Azul游戏渲染器，用于可视化游戏状态"""
    
    # 颜色定义
    COLORS = {
        PieceColor.BLUE: (100, 140, 255),
        PieceColor.YELLOW: (255, 230, 150),
        PieceColor.RED: (255, 130, 130),
        PieceColor.BLACK: (100, 100, 100),
        PieceColor.WHITE: (240, 240, 240),
        PieceColor.NONE: (200, 200, 200)
    }
    
    # 布局配置
    WINDOW_SIZE = (1200, 800)
    BOARD_SIZE = (400, 500)
    DISK_RADIUS = 50
    PIECE_RADIUS = 20
    CELL_SIZE = 40
    
    def __init__(self):
        """初始化渲染器"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Azul Game")
        self.font = pygame.font.Font(None, 36)
        
    def render(self, env: AzulEnv) -> Optional[pygame.Surface]:
        """
        渲染当前游戏状态
        
        Args:
            env: Azul游戏环境实例
            
        Returns:
            如果mode是'rgb_array'，返回RGB数组
        """
        # 清空屏幕
        self.screen.fill((255, 255, 255))
        
        # 渲染玩家棋盘
        self._render_board(env.player1_board, (50, 50))
        self._render_board(env.player2_board, (50, 400))
        
        # 渲染圆盘
        self._render_disks(env.disks)
        
        # 渲染待定区
        self._render_waiting_area(env.waiting_area)
        
        # 渲染当前玩家指示
        self._render_current_player(env.current_player)
        
        # 更新显示
        pygame.display.flip()
        
        return pygame.surfarray.array3d(self.screen)
        
    def _render_board(self, board: PlayerBoard, pos: Tuple[int, int]):
        """渲染单个玩家棋盘"""
        x, y = pos
        
        # 渲染玩家名称和分数
        text = f"{board.name}: {board.score}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (x, y - 30))
        
        # 渲染准备区
        for row in range(5):
            for col, piece in enumerate(board.prep_area[row]):
                if piece:
                    color = self.COLORS[piece.color]
                    center = (x + col * self.CELL_SIZE, y + row * self.CELL_SIZE)
                    pygame.draw.circle(self.screen, color, center, self.PIECE_RADIUS)
        
        # 渲染结算区
        score_x = x + 250
        for row in range(5):
            for col in range(5):
                piece = board.scoring_area[row][col]
                color = self.COLORS[piece.color if piece else PieceColor.NONE]
                center = (score_x + col * self.CELL_SIZE, y + row * self.CELL_SIZE)
                pygame.draw.circle(self.screen, color, center, self.PIECE_RADIUS)
        
        # 渲染扣分区
        penalty_y = y + 250
        for i, piece in enumerate(board.penalty_area):
            if piece:
                color = self.COLORS[piece.color]
                center = (x + i * self.CELL_SIZE, penalty_y)
                pygame.draw.circle(self.screen, color, center, self.PIECE_RADIUS)
                
    def _render_disks(self, disks: List[List[Piece]]):
        """渲染圆盘"""
        disk_x = 600
        disk_y = 100
        for i, disk in enumerate(disks):
            # 渲染圆盘背景
            center = (disk_x + (i % 3) * 150, disk_y + (i // 3) * 150)
            pygame.draw.circle(self.screen, (200, 200, 200), center, self.DISK_RADIUS)
            
            # 渲染圆盘上的棋子
            for j, piece in enumerate(disk):
                if piece:
                    angle = j * (360 / len(disk))
                    piece_x = center[0] + self.DISK_RADIUS * 0.6 * pygame.math.Vector2().from_polar((1, angle))[0]
                    piece_y = center[1] + self.DISK_RADIUS * 0.6 * pygame.math.Vector2().from_polar((1, angle))[1]
                    pygame.draw.circle(self.screen, self.COLORS[piece.color], (piece_x, piece_y), self.PIECE_RADIUS)
                    
    def _render_waiting_area(self, pieces: List[Piece]):
        """渲染待定区"""
        x = 600
        y = 400
        for i, piece in enumerate(pieces):
            if piece:
                color = self.COLORS[piece.color]
                center = (x + (i % 10) * self.CELL_SIZE, y + (i // 10) * self.CELL_SIZE)
                pygame.draw.circle(self.screen, color, center, self.PIECE_RADIUS)
                
    def _render_current_player(self, current_player: int):
        """渲染当前玩家指示器"""
        text = f"Current Player: {current_player}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (600, 50))
        
    def close(self):
        """关闭渲染器"""
        pygame.quit() 