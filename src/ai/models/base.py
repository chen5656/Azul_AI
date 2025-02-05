"""基础神经网络模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class AzulNet(nn.Module):
    """Azul游戏的神经网络模型基类"""
    
    def __init__(self, 
                 state_dim: int = 180,  # 状态空间维度
                 action_dim: int = 5,    # 动作空间维度
                 hidden_dim: int = 256   # 隐藏层维度
                 ):
        super().__init__()
        
        # 编码器：将状态转换为特征表示
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头：输出动作概率
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值头：输出状态价值估计
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 游戏状态张量 [batch_size, state_dim]
            
        Returns:
            action_probs: 动作概率分布 [batch_size, action_dim]
            state_value: 状态价值估计 [batch_size, 1]
        """
        features = self.encoder(state)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        
        return action_probs, state_value
        
    def save(self, path: str):
        """保存模型"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """加载模型"""
        self.load_state_dict(torch.load(path))
        
    @torch.no_grad()
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict]:
        """
        根据状态选择动作
        
        Args:
            state: 游戏状态数组
            deterministic: 是否使用确定性策略（选择概率最高的动作）
            
        Returns:
            action: 选择的动作
            info: 额外信息（如动作概率、状态价值等）
        """
        # 转换状态为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 获取动作概率和状态价值
        action_probs, state_value = self.forward(state_tensor)
        
        # 选择动作
        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            action = torch.multinomial(action_probs, 1).item()
            
        # 收集信息
        info = {
            'action_probs': action_probs.squeeze().numpy(),
            'state_value': state_value.item()
        }
        
        return action, info 