from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from .game_state import GameState
from .board import PlayerBoard
from .piece import Piece, PieceColor

class AzulEnv(gym.Env):
    """Azul游戏环境, 遵循OpenAI Gym风格的接口"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    # 动作空间定义
    # action = (source_type, source_idx, color, target_type, target_idx)
    # source_type: 0=圆盘, 1=待定区
    # source_idx: 圆盘/待定区的索引
    # color: 0=蓝, 1=黄, 2=红, 3=黑, 4=白
    # target_type: 0=准备区, 1=扣分区
    # target_idx: 目标行/列的索引
    
    # 类属性：所有实例共享
    PIECES_PER_COLOR = 20  # 每种颜色的瓷砖数量
    PIECES_PER_DISK = 4    # 每个圆盘的瓷砖数量
    
    def __init__(self):
        """构造方法：初始化实例"""
        super().__init__()  # 调用父类初始化
        
        # 定义动作空间和观察空间
        self.action_space = spaces.MultiDiscrete([2, 5, 5, 2, 5])  # (source_type, source_idx, color, target_type, target_idx)
        
        # 计算观察空间的维度
        obs_dim = 100 + 5 + 50 + 50 + 14 + 1 + 1  # 221维
        self.observation_space = spaces.Box(low=0, high=5, shape=(obs_dim,), dtype=np.float32)
        
        # 颜色映射
        self.COLORS = {
            'BLUE': 0,
            'YELLOW': 1,
            'RED': 2,
            'BLACK': 3,
            'WHITE': 4
        }
        
        # 初始化游戏状态
        self.state = GameState.INIT
        self.current_player = 1
        self.round_count = 0
        
        # 初始化棋盘
        self.player1_board = PlayerBoard(50, 50, "Player 1")
        self.player2_board = PlayerBoard(50, 400, "Player 2")
        
        # 初始化游戏组件
        self.piece_pool = self._initialize_piece_pool()
        self.waste_pool = []
        self.waiting_area = []
        self.disks = [[] for _ in range(5)]
        self.first_piece = None
        
    def reset(self) -> np.ndarray:
        """实例方法：重置环境"""
        self.__init__()
        self.state = GameState.RUNNING
        self.start_new_round()
        return self._get_observation()
        
    def step(self, action: Tuple[int, int, int, int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一个动作并返回结果
        
        Args:
            action: (source_type, source_idx, color, target_type, target_idx)
            
        Returns:
            observation: 游戏状态的向量表示
            reward: 即时奖励值
            done: 游戏是否结束
            info: 额外信息
        """
        if not self._is_valid_action(action):
            return self._get_observation(), -10.0, False, {'error': 'Invalid action'}
            
        # 执行动作
        source_type, source_idx, color, target_type, target_idx = action
        reward = self._execute_action(source_type, source_idx, color, target_type, target_idx)
        
        # 检查是否需要结算
        if self._need_scoring():
            scoring_reward = self._perform_scoring()
            reward += scoring_reward
            
        # 检查游戏是否结束
        done = self._check_game_end()
        if done:
            reward += self._calculate_final_reward()
            
        # 切换玩家
        if not done:
            self.current_player = 3 - self.current_player
            
        return self._get_observation(), reward, done, self._get_info()
        
    def _get_observation(self) -> np.ndarray:
        """
        获取游戏状态的向量表示
        
        Returns:
            observation: 包含以下信息的numpy数组：
            - 圆盘状态 (5, 4, 5) - 5个圆盘，每个4个位置，5种颜色的one-hot编码
            - 待定区状态 (5,) - 每种颜色的数量
            - 玩家准备区状态 (2, 5, 5) - 2个玩家，5行，每行最多5个位置
            - 玩家结算区状态 (2, 5, 5) - 2个玩家，5x5的结算区
            - 玩家扣分区状态 (2, 7) - 2个玩家，每人7个扣分位置
            - 当前玩家 (1,) - 1或2
            - 先手标记 (1,) - 是否持有先手标记
        """
        # 1. 圆盘状态编码
        disks_state = np.zeros((5, 4, 5))  # 5个圆盘，每个4个位置，5种颜色
        for disk_idx, disk in enumerate(self.disks):
            for piece_idx, piece in enumerate(disk):
                if piece:
                    color_idx = self._get_color_index(piece.color)
                    disks_state[disk_idx, piece_idx, color_idx] = 1
                    
        # 2. 待定区状态编码
        waiting_state = np.zeros(5)  # 5种颜色
        for piece in self.waiting_area:
            if not piece.is_first:  # 不统计先手标记
                color_idx = self._get_color_index(piece.color)
                waiting_state[color_idx] += 1
                
        # 3. 玩家准备区状态编码
        prep_state = np.zeros((2, 5, 5))  # 2个玩家，5行，每行最多5个位置
        # Player 1
        for row in range(5):
            for col, piece in enumerate(self.player1_board.prep_area[row]):
                if piece:
                    color_idx = self._get_color_index(piece.color)
                    prep_state[0, row, col] = color_idx + 1  # 使用1-5表示颜色，0表示空
        # Player 2
        for row in range(5):
            for col, piece in enumerate(self.player2_board.prep_area[row]):
                if piece:
                    color_idx = self._get_color_index(piece.color)
                    prep_state[1, row, col] = color_idx + 1
                    
        # 4. 玩家结算区状态编码
        wall_state = np.zeros((2, 5, 5))  # 2个玩家，5x5的结算区
        # Player 1
        for row in range(5):
            for col in range(5):
                piece = self.player1_board.scoring_area[row][col]
                if piece:
                    color_idx = self._get_color_index(piece.color)
                    wall_state[0, row, col] = color_idx + 1
        # Player 2
        for row in range(5):
            for col in range(5):
                piece = self.player2_board.scoring_area[row][col]
                if piece:
                    color_idx = self._get_color_index(piece.color)
                    wall_state[1, row, col] = color_idx + 1
                    
        # 5. 玩家扣分区状态编码
        penalty_state = np.zeros((2, 7))  # 2个玩家，每人7个扣分位置
        # Player 1
        for i, piece in enumerate(self.player1_board.penalty_area):
            if piece:
                penalty_state[0, i] = 1 if piece.is_first else 2  # 1表示先手标记，2表示普通棋子
        # Player 2
        for i, piece in enumerate(self.player2_board.penalty_area):
            if piece:
                penalty_state[1, i] = 1 if piece.is_first else 2
                
        # 6. 当前玩家编码
        current_player = np.array([self.current_player])
        
        # 7. 先手标记状态
        first_player_marker = np.array([1 if self.first_piece in self.waiting_area else 0])
        
        # 将所有状态拼接成一个大的状态向量
        observation = np.concatenate([
            disks_state.flatten(),          # 100 = 5 * 4 * 5
            waiting_state.flatten(),        # 5
            prep_state.flatten(),           # 50 = 2 * 5 * 5
            wall_state.flatten(),           # 50 = 2 * 5 * 5
            penalty_state.flatten(),        # 14 = 2 * 7
            current_player.flatten(),       # 1
            first_player_marker.flatten()   # 1
        ])
        
        return observation

    def _get_color_index(self, color: Tuple[int, int, int]) -> int:
        """将RGB颜色转换为索引"""
        # 假设color是RGB元组
        color_map = {
            (100, 140, 255): 0,  # BLUE
            (255, 230, 150): 1,  # YELLOW
            (255, 130, 130): 2,  # RED
            (100, 100, 100): 3,  # BLACK
            (240, 240, 240): 4   # WHITE
        }
        return color_map.get(color, 0)  # 默认返回0（蓝色）
        
    def _is_valid_action(self, action: Tuple[int, int, int, int, int]) -> bool:
        """检查动作是否合法"""
        source_type, source_idx, color, target_type, target_idx = action
        
        # 检查索引范围
        if source_type == 0:  # 圆盘
            if not (0 <= source_idx < 5):
                return False
        elif source_type == 1:  # 待定区
            if source_idx != 0:
                return False
        else:
            return False
            
        # 检查颜色是否有效
        if not (0 <= color < 5):
            return False
            
        # 检查目标位置
        if target_type == 0:  # 准备区
            if not (0 <= target_idx < 5):
                return False
        elif target_type == 1:  # 扣分区
            if target_idx != 0:
                return False
        else:
            return False
            
        return True
        
    def get_valid_actions(self) -> List[Tuple[int, int, int, int, int]]:
        """获取当前所有合法动作"""
        valid_actions = []
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 从圆盘选择
        for disk_idx, disk in enumerate(self.disks):
            colors_in_disk = set(piece.color for piece in disk)
            for color in colors_in_disk:
                # 可以放入准备区
                for row in range(5):
                    if current_board.can_place_pieces(row, color):
                        valid_actions.append((0, disk_idx, color, 0, row))
                # 总是可以放入扣分区
                valid_actions.append((0, disk_idx, color, 1, 0))
                
        # 从待定区选择
        colors_in_waiting = set(piece.color for piece in self.waiting_area if not piece.is_first)
        for color in colors_in_waiting:
            # 可以放入准备区
            for row in range(5):
                if current_board.can_place_pieces(row, color):
                    valid_actions.append((1, 0, color, 0, row))
            # 总是可以放入扣分区
            valid_actions.append((1, 0, color, 1, 0))
            
        return valid_actions
        
    def _execute_action(self, source_type: int, source_idx: int, color: int, 
                       target_type: int, target_idx: int) -> float:
        """
        执行动作并返回即时奖励
        
        Args:
            source_type (int): 源位置类型 (0=圆盘, 1=待定区)
            source_idx (int): 源位置索引
            color (int): 选择的颜色
            target_type (int): 目标位置类型 (0=准备区, 1=扣分区)
            target_idx (int): 目标位置索引
            
        Returns:
            float: 即时奖励值
        """
        # 获取当前玩家的棋盘
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 获取要移动的瓷砖
        pieces_to_move = []
        other_pieces = []
        source_pieces = self.disks[source_idx] if source_type == 0 else self.waiting_area
        
        # 从源位置收集瓷砖
        for piece in source_pieces[:]:
            if piece.is_first:  # 如果是先手标记
                pieces_to_move.append(piece)
                source_pieces.remove(piece)
            elif piece.color == PieceColor(color):  # 如果是选中的颜色
                pieces_to_move.append(piece)
                source_pieces.remove(piece)
            else:  # 其他颜色的瓷砖
                other_pieces.append(piece)
                source_pieces.remove(piece)
                
        # 如果是从圆盘取瓷砖，将其他颜色的瓷砖移到待定区
        if source_type == 0 and other_pieces:
            self.waiting_area.extend(other_pieces)
            
        # 处理瓷砖放置
        if target_type == 0:  # 放入准备区
            remaining = current_board.add_pieces(target_idx, pieces_to_move)
            if remaining:  # 如果有剩余瓷砖，放入扣分区
                current_board.add_to_penalty(remaining)
        else:  # 直接放入扣分区
            current_board.add_to_penalty(pieces_to_move)
            
        # 计算即时奖励（可以根据具体规则调整）
        reward = 0.0
        if target_type == 0:  # 放入准备区给予小额正奖励
            reward = 0.1
        else:  # 放入扣分区给予小额负奖励
            reward = -0.1
            
        return reward
        
    def _calculate_reward(self, action_result: Dict) -> float:
        """计算奖励值"""
        # 实现奖励计算逻辑
        pass 

    def render(self, mode='human'):
        """渲染当前游戏状态"""
        # TODO: 实现渲染逻辑
        pass
        
    def _initialize_piece_pool(self) -> List[Piece]:
        """
        初始化瓷砖池
        
        Returns:
            List[Piece]: 包含所有瓷砖的列表
        """
        piece_pool = []
        
        # 为每种颜色创建指定数量的瓷砖
        for color in PieceColor:
            if color != PieceColor.NONE:  # 跳过NONE颜色
                for _ in range(self.PIECES_PER_COLOR):
                    piece_pool.append(Piece(color=color))
                    
        # 创建先手标记
        self.first_piece = Piece.create_first_player_marker()
        
        # 随机打乱瓷砖顺序
        random.shuffle(piece_pool)
        
        return piece_pool
        
    def _need_scoring(self) -> bool:
        """检查是否需要进行结算"""
        # TODO: 实现结算检查
        pass
        
    def _perform_scoring(self) -> float:
        """执行结算并返回得分"""
        # TODO: 实现结算逻辑
        pass
        
    def _check_game_end(self) -> bool:
        """检查游戏是否结束"""
        # TODO: 实现游戏结束检查
        pass
        
    def _calculate_final_reward(self) -> float:
        """计算游戏结束时的最终奖励"""
        # TODO: 实现最终奖励计算
        pass
        
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'current_player': self.current_player,
            'round': self.round_count,
            'player1_score': self.player1_board.score,
            'player2_score': self.player2_board.score
        }
        
    def start_new_round(self):
        """
        开始新的回合
        
        1. 重置圆盘和待定区
        2. 将瓷砖分配到圆盘
        3. 处理剩余瓷砖
        4. 设置回合状态
        """
        # 重置游戏区域
        self.disks = [[] for _ in range(5)]  # 5个圆盘
        self.waiting_area = []
        
        # 如果是第一回合或瓷砖池为空，重新初始化瓷砖池
        if not self.piece_pool:
            self.piece_pool = self._initialize_piece_pool()
            
        # 将先手标记放入待定区（如果不在玩家手中）
        if self.first_piece.position is None:
            self.waiting_area.append(self.first_piece)
            
        # 将瓷砖分配到圆盘
        for disk_idx in range(5):
            pieces_to_add = min(self.PIECES_PER_DISK, len(self.piece_pool))
            if pieces_to_add == 0:
                break
                
            # 取出瓷砖并添加到圆盘
            disk_pieces = self.piece_pool[:pieces_to_add]
            self.piece_pool = self.piece_pool[pieces_to_add:]
            self.disks[disk_idx].extend(disk_pieces)
            
        # 增加回合计数
        self.round_count += 1
        
        # 设置游戏状态
        self.state = GameState.RUNNING
        
    def _execute_action(self, source_type: int, source_idx: int, color: int, 
                       target_type: int, target_idx: int) -> float:
        """
        执行动作并返回即时奖励
        
        Args:
            source_type (int): 源位置类型 (0=圆盘, 1=待定区)
            source_idx (int): 源位置索引
            color (int): 选择的颜色
            target_type (int): 目标位置类型 (0=准备区, 1=扣分区)
            target_idx (int): 目标位置索引
            
        Returns:
            float: 即时奖励值
        """
        # 获取当前玩家的棋盘
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 获取要移动的瓷砖
        pieces_to_move = []
        other_pieces = []
        source_pieces = self.disks[source_idx] if source_type == 0 else self.waiting_area
        
        # 从源位置收集瓷砖
        for piece in source_pieces[:]:
            if piece.is_first:  # 如果是先手标记
                pieces_to_move.append(piece)
                source_pieces.remove(piece)
            elif piece.color == PieceColor(color):  # 如果是选中的颜色
                pieces_to_move.append(piece)
                source_pieces.remove(piece)
            else:  # 其他颜色的瓷砖
                other_pieces.append(piece)
                source_pieces.remove(piece)
                
        # 如果是从圆盘取瓷砖，将其他颜色的瓷砖移到待定区
        if source_type == 0 and other_pieces:
            self.waiting_area.extend(other_pieces)
            
        # 处理瓷砖放置
        if target_type == 0:  # 放入准备区
            remaining = current_board.add_pieces(target_idx, pieces_to_move)
            if remaining:  # 如果有剩余瓷砖，放入扣分区
                current_board.add_to_penalty(remaining)
        else:  # 直接放入扣分区
            current_board.add_to_penalty(pieces_to_move)
            
        # 计算即时奖励（可以根据具体规则调整）
        reward = 0.0
        if target_type == 0:  # 放入准备区给予小额正奖励
            reward = 0.1
        else:  # 放入扣分区给予小额负奖励
            reward = -0.1
            
        return reward 