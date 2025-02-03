import unittest
import numpy as np
from game.environment import AzulEnv

class TestAzulEnvironment(unittest.TestCase):
    def setUp(self):
        """每个测试用例前都创建一个新的环境"""
        self.env = AzulEnv()
        
    def test_initial_state(self):
        """测试初始状态的正确性"""
        # 重置环境获取初始状态
        observation = self.env.reset()
        
        # 1. 检查向量维度
        self.assertEqual(len(observation), 221,  # 100 + 5 + 50 + 50 + 14 + 1 + 1
                        "状态向量维度应该是221")
        
        # 2. 检查圆盘状态 (前100个元素)
        disks_state = observation[:100].reshape(5, 4, 5)
        # 每个圆盘应该有4个棋子
        for disk in disks_state:
            self.assertEqual(np.sum(disk), 4,
                           "每个圆盘应该有4个棋子")
            
        # 3. 检查待定区状态 (接下来5个元素)
        waiting_state = observation[100:105]
        self.assertEqual(np.sum(waiting_state), 0,
                        "初始状态待定区应该是空的")
        
        # 4. 检查玩家准备区状态
        prep_state = observation[105:155].reshape(2, 5, 5)
        self.assertEqual(np.sum(prep_state), 0,
                        "初始状态准备区应该是空的")
        
        # 5. 检查玩家结算区状态
        wall_state = observation[155:205].reshape(2, 5, 5)
        self.assertEqual(np.sum(wall_state), 0,
                        "初始状态结算区应该是空的")
        
        # 6. 检查扣分区状态
        penalty_state = observation[205:219].reshape(2, 7)
        self.assertEqual(np.sum(penalty_state), 0,
                        "初始状态扣分区应该是空的")
        
        # 7. 检查当前玩家
        current_player = observation[219]
        self.assertEqual(current_player, 1,
                        "游戏应该从玩家1开始")
        
        # 8. 检查先手标记
        first_player_marker = observation[220]
        self.assertEqual(first_player_marker, 1,
                        "初始状态先手标记应该在待定区")
        
    def test_valid_actions(self):
        """测试合法动作生成"""
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        
        # 检查每个动作的格式和范围
        for action in valid_actions:
            self.assertEqual(len(action), 5,
                           "动作应该包含5个元素")
            source_type, source_idx, color, target_type, target_idx = action
            
            # 检查source_type
            self.assertIn(source_type, [0, 1],
                         "source_type应该是0或1")
            
            # 检查source_idx
            if source_type == 0:  # 圆盘
                self.assertIn(source_idx, range(5),
                            "圆盘索引应该在0-4范围内")
            else:  # 待定区
                self.assertEqual(source_idx, 0,
                               "待定区索引应该是0")
                
            # 检查color
            self.assertIn(color, range(5),
                         "颜色索引应该在0-4范围内")
                
            # 检查target_type和target_idx
            self.assertIn(target_type, [0, 1],
                         "target_type应该是0或1")
            if target_type == 0:  # 准备区
                self.assertIn(target_idx, range(5),
                            "准备区行索引应该在0-4范围内")
            else:  # 扣分区
                self.assertEqual(target_idx, 0,
                               "扣分区索引应该是0")
                
    def test_step_function(self):
        """测试执行动作后的状态变化"""
        self.env.reset()
        valid_actions = self.env.get_valid_actions()
        
        if valid_actions:  # 确保有合法动作
            action = valid_actions[0]
            next_state, reward, done, info = self.env.step(action)
            
            # 检查返回值的类型
            self.assertIsInstance(next_state, np.ndarray,
                                "next_state应该是numpy数组")
            self.assertIsInstance(reward, float,
                                "reward应该是浮点数")
            self.assertIsInstance(done, bool,
                                "done应该是布尔值")
            self.assertIsInstance(info, dict,
                                "info应该是字典")
            
            # 检查状态向量的维度
            self.assertEqual(len(next_state), 221,
                           "状态向量维度应该是221")
            
    def test_invalid_action(self):
        """测试无效动作的处理"""
        self.env.reset()
        
        # 创建一个明显无效的动作
        invalid_action = (9, 9, 9, 9, 9)
        next_state, reward, done, info = self.env.step(invalid_action)
        
        # 检查是否得到惩罚
        self.assertLess(reward, 0,
                       "无效动作应该得到负奖励")
        self.assertIn('error', info,
                     "info字典应该包含error信息")

if __name__ == '__main__':
    unittest.main() 