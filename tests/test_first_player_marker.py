import unittest
from src.game.environment import AzulEnv
from src.game.piece import Piece, PieceColor, PieceType
from src.game.game_state import GameState
import pytest

"""
测试,第一回合,第二回合,先手棋子开始都在待定区。 
如果待定区没有别的棋子,玩家不能选择先手棋子;如果待定区有别的棋子,玩家不能选择先手棋子。
如果待定区有棋子2个黑色,3个红色,以及先手棋子,准备区没有棋子,扣分区没有棋子,有三个并列的test:
    1.玩家选择2个黑色棋子后,待定区只剩2个红色棋子;
    2. 玩家选择2个黑色棋子后,把黑色棋子放入准备区第一排,那么扣分区会有一个黑色棋子和一个先手棋子;
    3. 玩家选择2个黑色棋子后,把黑色棋子放入扣分区,那么扣分区会有2个黑色棋子和一个先手棋子。
"""

class TestFirstPlayerMarker(unittest.TestCase):
    def setUp(self):
        """每个测试前的设置"""
        self.env = AzulEnv()
        self.env.reset()
        
    def test_first_player_marker_initial_position(self):
        """测试先手棋子的初始位置"""
        # 第一回合开始时
        self.assertTrue(any(p.is_first for p in self.env.waiting_area))
        
        # 执行一个回合后
        valid_actions = self.env.get_valid_actions()
        if valid_actions:
            self.env.step(valid_actions[0])
            self.env.start_new_round()
            # 第二回合开始时,先手棋子应该仍在待定区
            self.assertTrue(any(p.is_first for p in self.env.waiting_area))
            
    def test_cannot_select_lone_first_player_marker(self):
        """测试不能单独选择先手棋子"""
        # 清空待定区,只留下先手棋子
        self.env.waiting_area = [p for p in self.env.waiting_area if p.is_first]
        
        # 获取有效动作
        valid_actions = self.env.get_valid_actions()
        
        # 验证没有从待定区选择的动作
        self.assertFalse(any(action[0] == 1 for action in valid_actions))
        
    def test_waiting_area_interactions(self):
        """测试待定区的交互"""
        # 设置待定区状态：2个黑色,3个红色,1个先手棋子
        self.env.waiting_area = [
            Piece(PieceColor.BLACK) for _ in range(2)
        ] + [
            Piece(PieceColor.RED) for _ in range(3)
        ] + [
            Piece.create_first_player_marker()
        ]
        
        # 清空准备区和扣分区
        self.env.player1_board.preparation_area = [[] for _ in range(5)]
        self.env.player1_board.penalty_area = []
        
        # 测试1：选择黑色棋子后,待定区状态
        action_select_black = (1, 0, 3, 0, 0)  # 从待定区选择黑色棋子放入准备区第一行
        self.env.step(action_select_black)
        
        # 验证待定区只剩3个红色棋子
        remaining_pieces = [p for p in self.env.waiting_area if not p.is_first]
        self.assertEqual(len(remaining_pieces), 3)
        self.assertTrue(all(p.color == PieceColor.RED for p in remaining_pieces))
        
    def test_preparation_area_overflow(self):
        """测试准备区溢出情况"""
        # 设置相同的待定区状态
        self.env.waiting_area = [
            Piece(PieceColor.BLACK) for _ in range(2)
        ] + [
            Piece(PieceColor.RED) for _ in range(3)
        ] + [
            Piece.create_first_player_marker()
        ]
        
        # 清空准备区和扣分区
        self.env.player1_board.preparation_area = [[] for _ in range(5)]
        self.env.player1_board.penalty_area = []
        
        # 选择黑色棋子放入第一行（只能放1个）
        action = (1, 0, 3, 0, 0)  # 从待定区选择黑色棋子放入准备区第一行
        self.env.step(action)
        
        # 验证准备区和扣分区状态
        prep_area_pieces = self.env.player1_board.preparation_area[0]
        penalty_pieces = self.env.player1_board.penalty_area
        
        self.assertEqual(len(prep_area_pieces), 1)  # 准备区第一行应有1个黑色棋子
        self.assertEqual(len(penalty_pieces), 2)    # 扣分区应有1个黑色棋子和1个先手棋子
        
    def test_direct_to_penalty(self):
        """测试直接放入扣分区"""
        # 设置相同的待定区状态
        self.env.waiting_area = [
            Piece(PieceColor.BLACK) for _ in range(2)
        ] + [
            Piece(PieceColor.RED) for _ in range(3)
        ] + [
            Piece.create_first_player_marker()
        ]
        
        # 清空准备区和扣分区
        self.env.player1_board.preparation_area = [[] for _ in range(5)]
        self.env.player1_board.penalty_area = []
        
        # 选择黑色棋子直接放入扣分区
        action = (1, 0, 3, 1, 0)  # 从待定区选择黑色棋子放入扣分区
        self.env.step(action)
        
        # 验证扣分区状态
        penalty_pieces = self.env.player1_board.penalty_area
        
        self.assertEqual(len(penalty_pieces), 3)  # 扣分区应有2个黑色棋子和1个先手棋子
        black_pieces = [p for p in penalty_pieces if p.color == PieceColor.BLACK]
        first_player_pieces = [p for p in penalty_pieces if p.is_first]
        
        self.assertEqual(len(black_pieces), 2)
        self.assertEqual(len(first_player_pieces), 1)

if __name__ == '__main__':
    unittest.main() 