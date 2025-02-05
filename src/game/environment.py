from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from .game_state import GameState
from .board import PlayerBoard
from .piece import Piece, PieceColor
from .config import GameConfig

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
    
    def __init__(self, config: GameConfig = None):
        """构造方法：初始化实例"""
        super().__init__()
        
        # 使用配置或默认值
        self.config = config or GameConfig()
        
        # 修改动作空间以使用配置
        self.action_space = spaces.MultiDiscrete([
            2,  # source_type: 0=圆盘, 1=待定区
            self.config.NUM_DISKS,  # source_idx
            5,  # color
            2,  # target_type: 0=准备区, 1=扣分区
            5   # target_idx
        ])
        
        # 计算观察空间的维度
        disk_dim = self.config.NUM_DISKS * 5  # 圆盘状态：NUM_DISKS个圆盘 × 5种颜色
        waiting_dim = 5  # 待定区状态：5种颜色的数量
        prep_dim = self.config.NUM_PLAYERS * self.config.BOARD_SIZE * self.config.BOARD_SIZE  # 准备区状态
        score_dim = self.config.NUM_PLAYERS * self.config.BOARD_SIZE * self.config.BOARD_SIZE  # 结算区状态
        penalty_dim = self.config.NUM_PLAYERS * self.config.PENALTY_SLOTS  # 扣分区状态
        player_dim = 1  # 当前玩家
        first_marker_dim = 1  # 先手标记位置
        
        obs_dim = disk_dim + waiting_dim + prep_dim + score_dim + penalty_dim + player_dim + first_marker_dim
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
        self.player1_board = PlayerBoard("Player 1")
        self.player2_board = PlayerBoard("Player 2")
        
        # 初始化游戏组件
        self.piece_pool = []
        self.waste_pool = []
        self.waiting_area = []
        self.disks = [[] for _ in range(self.config.NUM_DISKS)]
        
        # 创建并放置先手标记到待定区
        self.first_piece = Piece.create_first_player_marker()
        self.waiting_area.append(self.first_piece)
        
        # 初始化其他组件
        self._initialize_game()
        
        # 初始化渲染器
        self.renderer = None
        
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
        self._execute_action(source_type, source_idx, color, target_type, target_idx)
        
        # 计算动作奖励
        action_result = {
            'target_type': target_type,
            'is_valid': True,
            'is_opponent': False  # 当前玩家的动作
        }
        reward = self._calculate_reward(action_result)
        
        # 检查是否需要结算
        if self._need_scoring():
            scoring_result = {
                'score': self._perform_scoring(),
                'is_opponent': False  # 当前玩家的得分
            }
            reward += self._calculate_reward(scoring_result)
            
        # 检查游戏是否结束
        done = self._check_game_end()
        if done:
            final_result = {
                'score': self._calculate_final_reward(),
                'is_opponent': False  # 当前玩家的最终得分
            }
            reward += self._calculate_reward(final_result)
            
        # 切换玩家
        if not done:
            self.current_player = 3 - self.current_player
            
        return self._get_observation(), reward, done, self._get_info()
        
    def _get_observation(self) -> np.ndarray:
        """
        获取当前状态的观察向量
        
        Returns:
            np.ndarray: 状态向量
            - 圆盘状态: NUM_DISKS个圆盘 × 5种颜色
            - 待定区状态: 5种颜色的数量
            - 准备区状态: NUM_PLAYERS × BOARD_SIZE × BOARD_SIZE
            - 结算区状态: NUM_PLAYERS × BOARD_SIZE × BOARD_SIZE
            - 扣分区状态: NUM_PLAYERS × PENALTY_SLOTS
            - 当前玩家: 1维
            - 先手标记位置: 1维
        """
        # 计算各部分的维度
        disk_dim = self.config.NUM_DISKS * 5
        waiting_dim = 5
        prep_dim = self.config.NUM_PLAYERS * self.config.BOARD_SIZE * self.config.BOARD_SIZE
        score_dim = self.config.NUM_PLAYERS * self.config.BOARD_SIZE * self.config.BOARD_SIZE
        penalty_dim = self.config.NUM_PLAYERS * self.config.PENALTY_SLOTS
        
        # 初始化状态向量
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # 1. 圆盘状态
        disk_idx = 0
        for disk in self.disks:
            for piece in disk:
                if piece and not piece.is_first:
                    color_idx = piece.color.id
                    obs[disk_idx * 5 + color_idx] += 1
            disk_idx += 1
        
        # 2. 待定区状态
        offset = disk_dim
        for piece in self.waiting_area:
            if not piece.is_first:
                color_idx = piece.color.id
                obs[offset + color_idx] += 1
        
        # 3. 准备区状态
        offset = disk_dim + waiting_dim
        for player_idx, board in enumerate([self.player1_board, self.player2_board]):
            for row in range(self.config.BOARD_SIZE):
                for col, piece in enumerate(board.prep_area[row]):
                    if piece and not piece.is_first:
                        color_idx = piece.color.id
                        obs[offset + player_idx * (self.config.BOARD_SIZE * self.config.BOARD_SIZE) + row * self.config.BOARD_SIZE + col] = color_idx + 1
        
        # 4. 结算区状态
        offset = disk_dim + waiting_dim + prep_dim
        for player_idx, board in enumerate([self.player1_board, self.player2_board]):
            for row in range(self.config.BOARD_SIZE):
                for col in range(self.config.BOARD_SIZE):
                    piece = board.scoring_area[row][col]
                    if piece and not piece.is_first:
                        color_idx = piece.color.id
                        obs[offset + player_idx * (self.config.BOARD_SIZE * self.config.BOARD_SIZE) + row * self.config.BOARD_SIZE + col] = color_idx + 1
        
        # 5. 扣分区状态
        offset = disk_dim + waiting_dim + prep_dim + score_dim
        for player_idx, board in enumerate([self.player1_board, self.player2_board]):
            for slot_idx, piece in enumerate(board.penalty_area):
                if piece and not piece.is_first:
                    color_idx = piece.color.id
                    obs[offset + player_idx * self.config.PENALTY_SLOTS + slot_idx] = color_idx + 1
        
        # 6. 当前玩家
        offset = disk_dim + waiting_dim + prep_dim + score_dim + penalty_dim
        obs[offset] = self.current_player
        
        # 7. 先手标记位置
        offset = disk_dim + waiting_dim + prep_dim + score_dim + penalty_dim + 1
        if any(p.is_first for p in self.waiting_area):
            obs[offset] = 1
        elif any(p.is_first for p in self.player1_board.penalty_area):
            obs[offset] = 2
        elif any(p.is_first for p in self.player2_board.penalty_area):
            obs[offset] = 3
        
        return obs

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
        
    def _calculate_reward(self, action_result: Dict) -> float:
        """
        计算动作的奖励值
        
        奖励规则：
        对于当前玩家：
        1. 成功放置到准备区：+0.1
        2. 放置到扣分区：-0.1
        3. 非法动作：-1.0
        4. 结算时的得分：实际得分
        
        对于对手动作：
        1. 对手成功放置到准备区：-0.1
        2. 对手放置到扣分区：+0.1
        3. 对手结算时的得分：-实际得分
        
        Args:
            action_result (Dict): 动作执行的结果，包含：
                - score: 得分（如果有结算）
                - is_valid: 动作是否合法
                - target_type: 目标位置类型
                - is_opponent: 是否是对手的动作
                
        Returns:
            float: 奖励值
        """
        # 获取动作是否来自对手
        is_opponent = action_result.get('is_opponent', False)
        multiplier = -1 if is_opponent else 1
        
        # 如果是非法动作
        if not action_result.get('is_valid', True):
            return -1.0 * multiplier
            
        # 如果有结算分数
        if 'score' in action_result:
            return float(action_result['score']) * multiplier
            
        # 根据放置位置给予基础奖励
        target_type = action_result.get('target_type', 0)
        base_reward = 0.1 if target_type == 0 else -0.1
        return base_reward * multiplier

    def render(self, mode='human'):
        """
        渲染当前游戏状态
        
        Args:
            mode (str): 渲染模式
                - 'human': 在屏幕上显示
                - 'rgb_array': 返回RGB数组
        
        Returns:
            numpy.ndarray: 如果mode是'rgb_array'，返回RGB数组
        """
        if self.renderer is None:
            from .renderer import AzulRenderer
            self.renderer = AzulRenderer()
            
        return self.renderer.render(self)
        
    def close(self):
        """关闭环境"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        
    def _initialize_piece_pool(self) -> List[Piece]:
        """
        初始化瓷砖池
        
        Returns:
            List[Piece]: 包含所有瓷砖的列表
        """
        piece_pool = []
        
        # 为每种颜色创建指定数量的瓷砖
        for color in list(PieceColor):  # 转换为列表
            if color != PieceColor.NONE:  # 跳过NONE颜色
                for _ in range(self.PIECES_PER_COLOR):
                    piece_pool.append(Piece(color=color))
                    
        # 随机打乱瓷砖顺序
        random.shuffle(piece_pool)
        
        return piece_pool
        
    def _need_scoring(self) -> bool:
        """
        检查是否需要进行结算
        
        结算条件：
        1. 所有圆盘为空
        2. 待定区为空
        3. 当前玩家的准备区有棋子
        
        Returns:
            bool: 如果需要结算返回True，否则返回False
        """
        # 获取当前玩家的棋盘
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 检查是否所有圆盘为空
        if any(disk for disk in self.disks):
            return False
            
        # 检查待定区是否为空
        if self.waiting_area:
            return False
            
        # 检查当前玩家的准备区是否有棋子
        has_pieces = False
        for row in current_board.prep_area:
            if any(piece is not None for piece in row):
                has_pieces = True
                break
                
        if has_pieces:
            self.state = GameState.SCORING
            return True
            
        return False
        
    def _perform_scoring(self) -> float:
        """
        执行结算并返回得分
        
        结算流程：
        1. 将当前玩家准备区的棋子移动到结算区
        2. 计算回合得分
        3. 清空准备区
        4. 将扣分区的棋子移到废弃池
        
        Returns:
            float: 结算得到的分数
        """
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 遍历准备区的每一行
        for row_idx, row in enumerate(current_board.prep_area):
            # 检查是否有完整的一行
            if all(piece is not None for piece in row):
                # 获取棋子颜色
                color = row[0].color
                # 找到结算区对应行中这个颜色应该放的位置
                target_col = current_board.SCORING_PATTERN[row_idx].index(color)
                # 将棋子放入结算区
                piece = row[0]
                piece.is_new = True  # 标记为新放置的棋子
                current_board.scoring_area[row_idx][target_col] = piece
                # 清空准备区这一行
                current_board.prep_area[row_idx] = [None] * len(row)
        
        # 计算回合得分
        round_score = current_board.calculate_round_end_score()
        
        # 处理扣分区
        pieces = current_board.clear_penalty_area()
        self.waste_pool.extend(pieces)
        
        # 更新游戏状态
        self.state = GameState.ROUND_END
        
        return float(round_score)
        
    def _check_game_end(self) -> bool:
        """
        检查游戏是否结束
        
        游戏结束条件：
        1. 任意玩家在结算区完成一整行
        2. 回合结束且无法开始新回合（可选）
        
        Returns:
            bool: 如果游戏结束返回True，否则返回False
        """
        # 检查玩家1的结算区
        for row in range(5):
            row_complete = True
            for col in range(5):
                if not self.player1_board.scoring_area[row][col]:
                    row_complete = False
                    break
            if row_complete:
                self.state = GameState.GAME_END
                return True
            
        # 检查玩家2的结算区
        for row in range(5):
            row_complete = True
            for col in range(5):
                if not self.player2_board.scoring_area[row][col]:
                    row_complete = False
                    break
            if row_complete:
                self.state = GameState.GAME_END
                return True
            
        # 检查是否还能开始新回合
        if (not self.piece_pool and  # 瓷砖池为空
            not any(self.disks) and  # 所有圆盘为空
            not self.waiting_area):  # 待定区为空
            self.state = GameState.GAME_END
            return True
        
        return False
        
    def _calculate_final_reward(self) -> float:
        """
        计算游戏结束时的最终奖励
        
        最终奖励包括：
        1. 完整行奖励：每行2分
        2. 完整列奖励：每列7分
        3. 颜色完成奖励：每种颜色10分
        
        Returns:
            float: 最终奖励值
        """
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 计算游戏结束时的额外得分
        final_score = current_board.calculate_game_end_score()
        
        # 更新玩家总分
        current_board.score += final_score
        
        return float(final_score)
        
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'current_player': self.current_player,
            'round': self.round_count,
            'player1_score': self.player1_board.score,
            'player2_score': self.player2_board.score
        }
        
    def start_new_round(self):
        """开始新回合"""
        # 检查棋子池是否需要补充
        if not self.piece_pool:
            self.piece_pool.extend(self.waste_pool)
            self.waste_pool.clear()
            random.shuffle(self.piece_pool)
            
        # 分配棋子到圆盘
        for disk_idx in range(self.config.NUM_DISKS):
            pieces_available = len(self.piece_pool)
            if pieces_available == 0:
                break
                
            # 计算这个圆盘可以放多少棋子
            pieces_to_add = min(self.config.PIECES_PER_DISK, pieces_available)
            
            # 取出棋子并添加到圆盘
            disk_pieces = self.piece_pool[:pieces_to_add]
            self.piece_pool = self.piece_pool[pieces_to_add:]
            self.disks[disk_idx].extend(disk_pieces)
            
        self.round_count += 1
        self.state = GameState.RUNNING
        
    def _initialize_game(self):
        """初始化游戏状态"""
        self.state = GameState.INIT
        self.current_player = 1
        self.round_count = 0
        self.piece_pool = self._initialize_piece_pool()
        
    def _execute_action(self, source_type: int, source_idx: int, color: int, 
                       target_type: int, target_idx: int):
        """
        执行动作
        
        Args:
            source_type (int): 源位置类型 (0=圆盘, 1=待定区)
            source_idx (int): 源位置索引
            color (int): 选择的颜色
            target_type (int): 目标位置类型 (0=准备区, 1=扣分区)
            target_idx (int): 目标位置索引
        """
        current_board = self.player1_board if self.current_player == 1 else self.player2_board
        
        # 获取要移动的棋子
        pieces_to_move = []
        other_pieces = []
        source_pieces = self.disks[source_idx] if source_type == 0 else self.waiting_area
        
        # 从源位置收集棋子
        for piece in source_pieces[:]:
            if piece.color == PieceColor(color) or piece.is_first:
                pieces_to_move.append(piece)
                source_pieces.remove(piece)
            else:
                other_pieces.append(piece)
                source_pieces.remove(piece)
                
        # 如果是从圆盘取棋子，将其他颜色的棋子移到待定区
        if source_type == 0 and other_pieces:
            self.waiting_area.extend(other_pieces)
            
        # 处理棋子放置
        if target_type == 0:  # 放入准备区
            # 确保从右到左填充
            row = current_board.prep_area[target_idx]
            start_idx = len(row) - 1
            
            # 检查是否可以放置
            if not current_board.can_place_pieces(target_idx, pieces_to_move):
                return  # 非法动作，直接返回
                
            # 分离先手标记和普通棋子
            normal_pieces = [p for p in pieces_to_move if not p.is_first]
            first_piece = next((p for p in pieces_to_move if p.is_first), None)
            
            # 放置普通棋子
            remaining = current_board.add_pieces_from_right(target_idx, normal_pieces)
            if remaining:
                current_board.add_to_penalty(remaining)
                
            # 处理先手标记
            if first_piece:
                current_board.add_to_penalty([first_piece])
                
        else:  # 直接放入扣分区
            current_board.add_to_penalty(pieces_to_move) 