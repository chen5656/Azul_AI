"""卷积神经网络模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from .base import AzulNet

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class AzulConvNet(AzulNet):
    """使用卷积神经网络的Azul模型"""
    
    def __init__(self,
                 board_size: int = 5,      # 棋盘大小
                 num_colors: int = 5,      # 颜色数量
                 num_channels: int = 128,  # 卷积通道数
                 num_blocks: int = 10      # 残差块数量
                 ):
        super().__init__()
        
        self.board_size = board_size
        self.num_colors = num_colors
        
        # 输入处理
        # 1. 准备区状态: [batch_size, num_colors, board_size, board_size]
        # 2. 结算区状态: [batch_size, num_colors, board_size, board_size]
        # 3. 圆盘状态: [batch_size, num_colors, num_disks]
        # 4. 待定区状态: [batch_size, num_colors]
        # 5. 扣分区状态: [batch_size, num_colors]
        
        # 主干网络
        self.conv_in = ConvBlock(num_colors * 2, num_channels)  # 准备区和结算区
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # 全局信息处理（圆盘、待定区、扣分区）
        self.global_info = nn.Sequential(
            nn.Linear(num_colors * 7, num_channels),  # 5个圆盘 + 待定区 + 扣分区
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(num_channels * 2, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels)
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, board_size * board_size * num_colors),
            nn.Softmax(dim=-1)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.ReLU(),
            nn.Linear(num_channels // 2, 1),
            nn.Tanh()  # 将值限制在[-1, 1]范围内
        )
        
    def _process_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理输入状态
        
        Args:
            state: 原始状态张量 [batch_size, state_dim]
            
        Returns:
            board_state: 棋盘状态 [batch_size, num_colors*2, board_size, board_size]
            global_state: 全局状态 [batch_size, num_colors*7]
        """
        batch_size = state.shape[0]
        
        # 解析状态向量
        board_state = state[:, :self.num_colors*2*self.board_size*self.board_size]
        global_state = state[:, self.num_colors*2*self.board_size*self.board_size:]
        
        # 重塑棋盘状态
        board_state = board_state.view(batch_size, self.num_colors*2, self.board_size, self.board_size)
        
        return board_state, global_state
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 游戏状态张量 [batch_size, state_dim]
            
        Returns:
            action_probs: 动作概率分布 [batch_size, action_dim]
            state_value: 状态价值估计 [batch_size, 1]
        """
        # 处理输入状态
        board_state, global_state = self._process_state(state)
        
        # 处理棋盘状态
        x = self.conv_in(board_state)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 全局平均池化
        board_features = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        
        # 处理全局信息
        global_features = self.global_info(global_state)
        
        # 特征融合
        features = self.fusion(torch.cat([board_features, global_features], dim=-1))
        
        # 输出动作概率和状态价值
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        
        return action_probs, state_value
        
    @torch.no_grad()
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        获取所有合法动作的概率分布
        
        Args:
            state: 游戏状态数组
            
        Returns:
            action_probs: 动作概率分布数组
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.forward(state_tensor)
        return action_probs.squeeze().numpy()
        
    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """
        获取状态价值估计
        
        Args:
            state: 游戏状态数组
            
        Returns:
            value: 状态价值估计
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, state_value = self.forward(state_tensor)
        return state_value.item() 