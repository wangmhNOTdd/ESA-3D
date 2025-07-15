#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum
from typing import Optional, Tuple


class RadialBasisFunction(nn.Module):
    """径向基函数，用于编码距离信息"""
    
    def __init__(self, num_radial: int = 64, cutoff: float = 10.0):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        
        # 高斯径向基函数的中心和宽度
        self.centers = nn.Parameter(torch.linspace(0, cutoff, num_radial))
        self.widths = nn.Parameter(torch.ones(num_radial) * 0.5)
        
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [N, ...] 距离张量
        Returns:
            [N, ..., num_radial] 径向基函数特征
        """
        # 将distances扩展为 [N, ..., num_radial]
        distances = distances.unsqueeze(-1)
        centers = self.centers.view(1, -1)
        widths = self.widths.view(1, -1)
        
        # 计算高斯径向基函数
        rbf = torch.exp(-widths * (distances - centers) ** 2)
        
        # 应用平滑截断
        cutoff_mask = distances <= self.cutoff
        rbf = rbf * cutoff_mask.float()
        
        return rbf


class EquivariantEdgeAttention(nn.Module):
    """等变边注意力模块 - ESA-3D的核心组件"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff
        self.activation = activation
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        
        # 径向基函数
        self.rbf = RadialBasisFunction(num_radial, cutoff)
        
        # 不变特征的Query, Key, Value映射
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 注意力权重计算的MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.head_dim * 2 + num_radial + 1, hidden_dim),  # q, k, rbf, dot_product
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_heads),
        )
        
        # 等变特征更新的门控机制
        self.coord_gate_mlp = nn.Sequential(
            nn.Linear(self.head_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 输出映射
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim] 边的不变特征
        edge_coords: torch.Tensor,    # [E, 3] 边的等变特征(相对位置向量)
        edge_mask: torch.Tensor,      # [E, E] 边-边连接的掩码
        edge_index: torch.Tensor,     # [2, E] 边的索引 (用于debug)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对边进行等变注意力更新
        
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_mask: [E, E] 边-边连接掩码，0表示连接，-inf表示不连接
            edge_index: [2, E] 边的索引
            
        Returns:
            updated_edge_features: [E, hidden_dim] 更新后的边不变特征
            updated_edge_coords: [E, 3] 更新后的边等变特征
        """
        E = edge_features.shape[0]
        
        # 生成Query, Key, Value
        q = self.q_proj(edge_features)  # [E, hidden_dim]
        k = self.k_proj(edge_features)  # [E, hidden_dim]
        v = self.v_proj(edge_features)  # [E, hidden_dim]
        
        # 重塑为多头形式
        q = q.reshape(E, self.num_heads, self.head_dim)  # [E, num_heads, head_dim]
        k = k.reshape(E, self.num_heads, self.head_dim)  # [E, num_heads, head_dim]
        v = v.reshape(E, self.num_heads, self.head_dim)  # [E, num_heads, head_dim]
        
        # 计算几何信息
        coord_diff = edge_coords.unsqueeze(1) - edge_coords.unsqueeze(0)  # [E, E, 3]
        distances = torch.norm(coord_diff, dim=-1)  # [E, E]
        
        # 添加小的epsilon避免除零
        distances = distances + 1e-8
        
        dot_products = torch.sum(
            edge_coords.unsqueeze(1) * edge_coords.unsqueeze(0), dim=-1
        )  # [E, E]
        
        # 裁剪到合理范围
        distances = torch.clamp(distances, min=1e-8, max=1e8)
        dot_products = torch.clamp(dot_products, min=-1e8, max=1e8)
        
        # 计算径向基函数特征
        rbf_features = self.rbf(distances)  # [E, E, num_radial]
        
        # 为每个头计算注意力权重
        attention_weights = []
        for head in range(self.num_heads):
            # 获取当前头的q, k
            q_head = q[:, head, :]  # [E, head_dim]
            k_head = k[:, head, :]  # [E, head_dim]
            
            # 计算pairwise的注意力输入
            q_expanded = q_head.unsqueeze(1).expand(-1, E, -1)  # [E, E, head_dim]
            k_expanded = k_head.unsqueeze(0).expand(E, -1, -1)  # [E, E, head_dim]
            
            # 拼接所有特征
            attention_input = torch.cat([
                q_expanded,  # [E, E, head_dim]
                k_expanded,  # [E, E, head_dim]
                rbf_features,  # [E, E, num_radial]
                dot_products.unsqueeze(-1),  # [E, E, 1]
            ], dim=-1)  # [E, E, 2*head_dim + num_radial + 1]
            
            # 通过MLP计算注意力分数
            attention_scores = self.attention_mlp(attention_input)  # [E, E, num_heads]
            attention_scores = attention_scores[:, :, head]  # [E, E]
            
            # 应用mask
            attention_scores = attention_scores + edge_mask
            
            # 检查数值稳定性
            attention_scores = torch.clamp(attention_scores, min=-1e9, max=1e9)
            
            # Softmax归一化
            attention_weights_head = torch.softmax(attention_scores, dim=-1)  # [E, E]
            
            # 处理可能的NaN值
            attention_weights_head = torch.nan_to_num(attention_weights_head, nan=0.0, posinf=1.0, neginf=0.0)
            
            attention_weights.append(attention_weights_head)
        
        attention_weights = torch.stack(attention_weights, dim=0)  # [num_heads, E, E]
        
        # 聚合不变特征
        updated_features = []
        for head in range(self.num_heads):
            v_head = v[:, head, :]  # [E, head_dim]
            attention_head = attention_weights[head]  # [E, E]
            
            # 加权聚合
            updated_v = torch.matmul(attention_head, v_head)  # [E, head_dim]
            updated_features.append(updated_v)
        
        updated_features = torch.cat(updated_features, dim=-1)  # [E, hidden_dim]
        
        # 更新等变特征
        updated_coords = edge_coords.clone()
        for head in range(self.num_heads):
            attention_head = attention_weights[head]  # [E, E]
            v_head = v[:, head, :]  # [E, head_dim]
            
            # 计算门控信号
            gates = self.coord_gate_mlp(v_head)  # [E, 1]
            gates_expanded = gates.unsqueeze(1).expand(-1, E, -1)  # [E, E, 1]
            
            # 加权聚合相对位置向量
            weighted_coord_diff = (
                attention_head.unsqueeze(-1) * gates_expanded * coord_diff
            )  # [E, E, 3]
            
            coord_update = torch.sum(weighted_coord_diff, dim=1)  # [E, 3]
            updated_coords = updated_coords + coord_update / self.num_heads
        
        # 残差连接和归一化
        updated_features = self.norm(edge_features + self.out_proj(updated_features))
        
        return updated_features, updated_coords
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class ESA3DBlock(nn.Module):
    """ESA-3D的基本块，包含Intra-Block和Inter-Block注意力"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 区块内注意力
        self.intra_attention = EquivariantEdgeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            activation=activation,
            dropout=dropout,
        )
        
        # 区块间注意力
        self.inter_attention = EquivariantEdgeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            activation=activation,
            dropout=dropout,
        )
        
        # 等变前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            activation,
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            intra_mask: [E, E] 区块内连接掩码
            inter_mask: [E, E] 区块间连接掩码
            edge_index: [2, E] 边的索引
        """
        # 区块内注意力 (先向内看)
        intra_features, intra_coords = self.intra_attention(
            edge_features, edge_coords, intra_mask, edge_index
        )
        
        # 区块间注意力 (再向外看)
        inter_features, inter_coords = self.inter_attention(
            intra_features, intra_coords, inter_mask, edge_index
        )
        
        # 等变前馈网络
        ffn_features = self.ffn(inter_features)
        final_features = self.norm2(inter_features + self.dropout(ffn_features))
        
        return final_features, inter_coords


def create_intra_block_mask(
    edge_index: torch.Tensor,
    block_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    创建区块内掩码：只有同一区块内的边且共享节点的边之间才能通信
    
    Args:
        edge_index: [2, E] 边的索引
        block_ids: [N] 每个节点的区块ID
        device: 设备
    
    Returns:
        intra_mask: [E, E] 区块内掩码
    """
    E = edge_index.shape[1]
    
    if E > 10000:  # 对于大图，使用稀疏掩码
        # 只允许少量边进行注意力计算
        print(f"警告：边数量过多 ({E})，使用稀疏掩码")
        intra_mask = torch.full((E, E), -float('inf'), device=device)
        # 只为每条边找到最近的几条边
        for i in range(min(E, 1000)):  # 限制处理的边数
            intra_mask[i, i] = 0.0  # 自注意力
        return intra_mask
    
    # 获取每条边的起始和结束节点的区块ID
    edge_block_ids = block_ids[edge_index]  # [2, E]
    
    # 创建掩码矩阵
    intra_mask = torch.full((E, E), -float('inf'), device=device)
    
    # 向量化版本：检查边是否在同一区块且共享节点
    src_blocks = edge_block_ids[0]  # [E]
    dst_blocks = edge_block_ids[1]  # [E]
    
    # 使用广播创建掩码
    src_match = src_blocks.unsqueeze(0) == src_blocks.unsqueeze(1)  # [E, E]
    dst_match = dst_blocks.unsqueeze(0) == dst_blocks.unsqueeze(1)  # [E, E]
    
    same_block = src_match & dst_match
    
    # 检查是否共享节点
    src_nodes = edge_index[0]  # [E]
    dst_nodes = edge_index[1]  # [E]
    
    share_src = src_nodes.unsqueeze(0) == src_nodes.unsqueeze(1)  # [E, E]
    share_dst = dst_nodes.unsqueeze(0) == dst_nodes.unsqueeze(1)  # [E, E]
    share_cross1 = src_nodes.unsqueeze(0) == dst_nodes.unsqueeze(1)  # [E, E]
    share_cross2 = dst_nodes.unsqueeze(0) == src_nodes.unsqueeze(1)  # [E, E]
    
    share_node = share_src | share_dst | share_cross1 | share_cross2
    
    # 最终掩码：同一区块且共享节点
    final_mask = same_block & share_node
    intra_mask[final_mask] = 0.0
    
    return intra_mask


def create_inter_block_mask(
    edge_index: torch.Tensor,
    block_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    创建区块间掩码：只有不同区块的边之间才能通信
    
    Args:
        edge_index: [2, E] 边的索引
        block_ids: [N] 每个节点的区块ID
        device: 设备
    
    Returns:
        inter_mask: [E, E] 区块间掩码
    """
    E = edge_index.shape[1]
    
    if E > 10000:  # 对于大图，使用稀疏掩码
        print(f"警告：边数量过多 ({E})，使用稀疏掩码")
        inter_mask = torch.full((E, E), -float('inf'), device=device)
        # 只为每条边找到最近的几条边
        for i in range(min(E, 1000)):  # 限制处理的边数
            inter_mask[i, i] = 0.0  # 自注意力
        return inter_mask
    
    # 获取每条边的起始和结束节点的区块ID
    edge_block_ids = block_ids[edge_index]  # [2, E]
    
    # 创建掩码矩阵
    inter_mask = torch.full((E, E), -float('inf'), device=device)
    
    # 向量化版本：检查边是否在不同区块
    src_blocks = edge_block_ids[0]  # [E]
    dst_blocks = edge_block_ids[1]  # [E]
    
    # 使用广播创建掩码
    src_match = src_blocks.unsqueeze(0) == src_blocks.unsqueeze(1)  # [E, E]
    dst_match = dst_blocks.unsqueeze(0) == dst_blocks.unsqueeze(1)  # [E, E]
    
    same_block = src_match & dst_match
    
    # 区块间掩码：不在同一区块的边
    inter_mask[~same_block] = 0.0
    
    return inter_mask
