import unittest
from src.game.board import PlayerBoard
from src.game.piece import Piece, PieceColor
from src.game.config import GameConfig

class TestScoring(unittest.TestCase):
    def setUp(self):
        """
        初始化测试环境
        """
        self.board = PlayerBoard(0, 0, "Test Player")
        
    def set_board_state(self, matrix: list):
        """
        通过矩阵设置棋盘状态，颜色根据SCORING_PATTERN自动设置
        
        Args:
            matrix (list): 5x5的矩阵
                -1: 空位
                0: 旧棋子
                1: 新棋子
        """
        for row in range(5):
            for col in range(5):
                print(row, col)
                if row < len(matrix) and col < len(matrix[row]):
                    if matrix[row][col] == -1:
                        self.board.scoring_area[row][col] = None
                    else:
                        color = self.board.SCORING_PATTERN[row][col]
                        piece = Piece(color)
                        piece.is_new = (matrix[row][col] == 1)
                        self.board.scoring_area[row][col] = piece
                else:
                    self.board.scoring_area[row][col] = None

    def print_board_state(self):
        """
        打印当前棋盘状态
        """
        color_map = {
            PieceColor.BLUE: 'B',
            PieceColor.YELLOW: 'Y',
            PieceColor.RED: 'R',
            PieceColor.BLACK: 'K',
            PieceColor.WHITE: 'W',
            None: ' '
        }
        
        print("\n当前棋盘状态:")
        for row in self.board.scoring_area:
            print("[", end="")
            for piece in row:
                if piece is None:
                    print("[ ]", end="")
                else:
                    marker = '*' if piece.is_new else ' '
                    print(f"[{color_map[piece.color]}{marker}]", end="")
            print("]")

    def test_new_pieces_scoring(self):
        """
        测试只对新放置棋子计算分数
        """
        # 测试场景：L形状，部分是新棋子
        pattern = [
            [0, 0, 1, -1, -1],  # B Y R* - -  (*表示新棋子)
            [1, -1, -1, -1, -1],  # W* - - - -
            [0, -1, -1, -1, -1],  # K - - - -
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1]
        ]
        
        self.set_board_state(pattern)
        self.print_board_state()
        
        # 计算期望分数
        # 基础分：2个新棋子 = 2分
        # 连接分：2个连接（R与Y相连，W与B相连）= 2分
        expected_score = 6
        
        actual_score = self.board.calculate_round_end_score()
        print(f"期望得分: {expected_score}")
        print(f"实际得分: {actual_score}")
        self.assertEqual(actual_score, expected_score)
        
if __name__ == '__main__':
    unittest.main() 