#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
优化的等变边注意力模块
解决O(E²)内存瓶颈问题，支持大分子/大图的可扩展性
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum
from torch_geometric.utils import to_dense_batch
from typing import Optional, Tuple, List
import numpy as np


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
        cutoff_mask = distances.squeeze(-1) <= self.cutoff
        rbf = rbf * cutoff_mask.unsqueeze(-1).float()
        
        return rbf


class SparseEquivariantEdgeAttention(nn.Module):
    """稀疏等变边注意力模块 - 解决O(E²)内存问题"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        k_neighbors: int = 32,  # KNN近邻数量
        use_sparse: bool = True,  # 是否使用稀疏注意力
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff
        self.k_neighbors = k_neighbors
        self.use_sparse = use_sparse
        self.activation = activation
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0
        
        # 径向基函数
        self.rbf = RadialBasisFunction(num_radial, cutoff)
        
        # 不变特征的Query, Key, Value映射
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 注意力权重计算的MLP
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.head_dim * 2 + num_radial + 1, hidden_dim),
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
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_index: torch.Tensor,     # [2, E]
        node_coords: torch.Tensor,    # [N, 3]
        attention_mask: Optional[torch.Tensor] = None,  # [E, E] 注意力掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_index: [2, E] 边的索引
            node_coords: [N, 3] 节点坐标
            attention_mask: [E, E] 注意力掩码
        
        Returns:
            updated_features: [E, hidden_dim] 更新后的边不变特征
            updated_coords: [E, 3] 更新后的边等变特征
        """
        E = edge_features.shape[0]
        
        # 计算边的中心坐标 (边的两个端点的中点)
        edge_center_coords = (node_coords[edge_index[0]] + node_coords[edge_index[1]]) / 2.0
        
        # 使用稀疏注意力或全注意力
        if self.use_sparse and E > 64:  # 当边数较多时使用稀疏注意力
            return self._sparse_attention(
                edge_features, edge_coords, edge_center_coords, attention_mask
            )
        else:
            return self._full_attention(
                edge_features, edge_coords, edge_center_coords, attention_mask
            )
    
    def _sparse_attention(
        self,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        edge_center_coords: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """稀疏注意力机制"""
        E = edge_features.shape[0]
        
        # 1. 基于空间距离的KNN近邻
        distances = torch.cdist(edge_center_coords, edge_center_coords)  # [E, E]
        
        # 找到每个边的k个最近邻
        if E > self.k_neighbors:
            _, nearest_indices = torch.topk(distances, self.k_neighbors, dim=1, largest=False)
        else:
            nearest_indices = torch.arange(E, device=distances.device).unsqueeze(0).expand(E, -1)
        
        # 2. 应用注意力掩码
        if attention_mask is not None:
            # 只在允许的边对上进行注意力
            sparse_mask = torch.zeros_like(distances, dtype=torch.bool)
            for i in range(E):
                for j in nearest_indices[i]:
                    if attention_mask[i, j]:
                        sparse_mask[i, j] = True
        else:
            sparse_mask = torch.zeros_like(distances, dtype=torch.bool)
            for i in range(E):
                sparse_mask[i, nearest_indices[i]] = True
        
        # 3. 稀疏注意力计算
        updated_features = edge_features.clone()
        updated_coords = edge_coords.clone()
        
        # 计算Query, Key, Value
        q = self.q_proj(edge_features).view(E, self.num_heads, self.head_dim)
        k = self.k_proj(edge_features).view(E, self.num_heads, self.head_dim)
        v = self.v_proj(edge_features).view(E, self.num_heads, self.head_dim)
        
        # 计算相对位置
        coord_diff = edge_center_coords.unsqueeze(1) - edge_center_coords.unsqueeze(0)  # [E, E, 3]
        
        # 对每个头进行注意力计算
        for head in range(self.num_heads):
            q_head = q[:, head, :]  # [E, head_dim]
            k_head = k[:, head, :]  # [E, head_dim]
            v_head = v[:, head, :]  # [E, head_dim]
            
            # 只计算稀疏位置的注意力
            head_attention = torch.zeros(E, E, device=edge_features.device)
            
            for i in range(E):
                for j in nearest_indices[i]:
                    if sparse_mask[i, j]:
                        # 计算注意力权重
                        dist = torch.norm(coord_diff[i, j])
                        
                        if dist <= self.cutoff:
                            # 特征拼接
                            rbf_features = self.rbf(dist.unsqueeze(0))  # [1, num_radial]
                            dot_product = torch.dot(q_head[i], k_head[j]).unsqueeze(0)  # [1]
                            
                            attention_input = torch.cat([
                                q_head[i],  # [head_dim]
                                k_head[j],  # [head_dim]
                                rbf_features.squeeze(0),  # [num_radial]
                                dot_product,  # [1]
                            ])
                            
                            attention_weight = self.attention_mlp(attention_input)[head]
                            head_attention[i, j] = attention_weight
            
            # 归一化注意力权重
            head_attention = F.softmax(head_attention, dim=1)
            
            # 更新不变特征
            for i in range(E):
                weighted_v = torch.zeros_like(v_head[i])
                for j in nearest_indices[i]:
                    if sparse_mask[i, j]:
                        weighted_v += head_attention[i, j] * v_head[j]
                updated_features[i] += weighted_v / self.num_heads
            
            # 更新等变特征
            gates = self.coord_gate_mlp(v_head)  # [E, 1]
            for i in range(E):
                coord_update = torch.zeros(3, device=edge_coords.device)
                for j in nearest_indices[i]:
                    if sparse_mask[i, j]:
                        coord_update += head_attention[i, j] * gates[j, 0] * coord_diff[i, j]
                updated_coords[i] += coord_update / self.num_heads
        
        # 残差连接和归一化
        updated_features = self.norm(edge_features + self.out_proj(updated_features))
        
        return updated_features, updated_coords
    
    def _full_attention(
        self,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        edge_center_coords: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """全注意力机制（适用于小图）"""
        E = edge_features.shape[0]
        
        # 计算Query, Key, Value
        q = self.q_proj(edge_features).view(E, self.num_heads, self.head_dim)
        k = self.k_proj(edge_features).view(E, self.num_heads, self.head_dim)
        v = self.v_proj(edge_features).view(E, self.num_heads, self.head_dim)
        
        # 计算相对位置
        coord_diff = edge_center_coords.unsqueeze(1) - edge_center_coords.unsqueeze(0)  # [E, E, 3]
        distances = torch.norm(coord_diff, dim=2)  # [E, E]
        
        # 计算注意力权重
        attention_weights = []
        
        for head in range(self.num_heads):
            q_head = q[:, head, :]  # [E, head_dim]
            k_head = k[:, head, :]  # [E, head_dim]
            
            # 计算注意力分数
            attention_scores = torch.zeros(E, E, device=edge_features.device)
            
            for i in range(E):
                for j in range(E):
                    if i == j or distances[i, j] <= self.cutoff:
                        # 距离编码
                        rbf_features = self.rbf(distances[i, j].unsqueeze(0))  # [1, num_radial]
                        dot_product = torch.dot(q_head[i], k_head[j]).unsqueeze(0)  # [1]
                        
                        # 特征拼接
                        attention_input = torch.cat([
                            q_head[i],  # [head_dim]
                            k_head[j],  # [head_dim]
                            rbf_features.squeeze(0),  # [num_radial]
                            dot_product,  # [1]
                        ])
                        
                        attention_scores[i, j] = self.attention_mlp(attention_input)[head]
                    else:
                        attention_scores[i, j] = float('-inf')
            
            # 应用注意力掩码
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
            
            # 归一化
            attention_weights.append(F.softmax(attention_scores, dim=1))
        
        # 更新特征
        updated_features = torch.zeros_like(edge_features)
        
        for head in range(self.num_heads):
            attention_head = attention_weights[head]  # [E, E]
            v_head = v[:, head, :]  # [E, head_dim]
            
            # 加权聚合
            weighted_v = torch.matmul(attention_head, v_head)  # [E, head_dim]
            updated_features += weighted_v / self.num_heads
        
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


class BlockBasedAttention(nn.Module):
    """基于区块的注意力机制 - 进一步优化内存使用"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        block_size: int = 64,  # 区块大小
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff
        self.block_size = block_size
        self.activation = activation
        self.dropout = dropout
        
        assert hidden_dim % num_heads == 0
        
        # 复用SparseEquivariantEdgeAttention的组件
        self.sparse_attention = SparseEquivariantEdgeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            k_neighbors=min(32, block_size),
            use_sparse=True,
            activation=activation,
            dropout=dropout,
        )
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_index: torch.Tensor,     # [2, E]
        node_coords: torch.Tensor,    # [N, 3]
        attention_mask: Optional[torch.Tensor] = None,  # [E, E] 注意力掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分块处理大图的边注意力
        """
        E = edge_features.shape[0]
        
        # 如果边数小于区块大小，直接使用稀疏注意力
        if E <= self.block_size:
            return self.sparse_attention(
                edge_features, edge_coords, edge_index, node_coords, attention_mask
            )
        
        # 分块处理
        updated_features = torch.zeros_like(edge_features)
        updated_coords = torch.zeros_like(edge_coords)
        
        num_blocks = (E + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min((block_idx + 1) * self.block_size, E)
            
            # 当前区块的边
            block_edge_features = edge_features[start_idx:end_idx]
            block_edge_coords = edge_coords[start_idx:end_idx]
            block_edge_index = edge_index[:, start_idx:end_idx]
            
            # 当前区块的注意力掩码
            block_attention_mask = None
            if attention_mask is not None:
                block_attention_mask = attention_mask[start_idx:end_idx, start_idx:end_idx]
            
            # 应用稀疏注意力
            block_updated_features, block_updated_coords = self.sparse_attention(
                block_edge_features, block_edge_coords, block_edge_index, 
                node_coords, block_attention_mask
            )
            
            # 更新结果
            updated_features[start_idx:end_idx] = block_updated_features
            updated_coords[start_idx:end_idx] = block_updated_coords
        
        return updated_features, updated_coords


class OptimizedESA3DBlock(nn.Module):
    """优化的ESA-3D块，支持多种注意力机制"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        attention_type: str = 'sparse',  # 'sparse', 'block', 'full'
        k_neighbors: int = 32,
        block_size: int = 64,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention_type = attention_type
        
        if attention_type == 'sparse':
            self.intra_attention = SparseEquivariantEdgeAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                k_neighbors=k_neighbors,
                use_sparse=True,
                activation=activation,
                dropout=dropout,
            )
            self.inter_attention = SparseEquivariantEdgeAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                k_neighbors=k_neighbors,
                use_sparse=True,
                activation=activation,
                dropout=dropout,
            )
        elif attention_type == 'block':
            self.intra_attention = BlockBasedAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                block_size=block_size,
                activation=activation,
                dropout=dropout,
            )
            self.inter_attention = BlockBasedAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                block_size=block_size,
                activation=activation,
                dropout=dropout,
            )
        else:  # full
            self.intra_attention = SparseEquivariantEdgeAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                k_neighbors=k_neighbors,
                use_sparse=False,
                activation=activation,
                dropout=dropout,
            )
            self.inter_attention = SparseEquivariantEdgeAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                k_neighbors=k_neighbors,
                use_sparse=False,
                activation=activation,
                dropout=dropout,
            )
        
        # 交错注意力的融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.coord_fusion = nn.Linear(6, 3)  # 融合两个坐标更新
        
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_index: torch.Tensor,     # [2, E]
        node_coords: torch.Tensor,    # [N, 3]
        intra_mask: torch.Tensor,     # [E, E] 区块内掩码
        inter_mask: torch.Tensor,     # [E, E] 区块间掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_index: [2, E] 边的索引
            node_coords: [N, 3] 节点坐标
            intra_mask: [E, E] 区块内注意力掩码
            inter_mask: [E, E] 区块间注意力掩码
        
        Returns:
            final_edge_features: [E, hidden_dim] 最终的边不变特征
            final_edge_coords: [E, 3] 最终的边等变特征
        """
        
        # 区块内注意力
        intra_features, intra_coords = self.intra_attention(
            edge_features, edge_coords, edge_index, node_coords, intra_mask
        )
        
        # 区块间注意力
        inter_features, inter_coords = self.inter_attention(
            edge_features, edge_coords, edge_index, node_coords, inter_mask
        )
        
        # 融合不变特征
        combined_features = torch.cat([intra_features, inter_features], dim=1)
        final_edge_features = self.fusion_mlp(combined_features)
        
        # 融合等变特征
        combined_coords = torch.cat([intra_coords, inter_coords], dim=1)
        final_edge_coords = self.coord_fusion(combined_coords)
        
        return final_edge_features, final_edge_coords


# 掩码生成函数
def create_intra_block_mask(
    edge_index: torch.Tensor, 
    block_ids: torch.Tensor, 
    device: torch.device,
    use_sparse: bool = False
) -> torch.Tensor:
    """创建区块内掩码 - 优化版本"""
    E = edge_index.shape[1]
    
    if use_sparse:
        # 使用稀疏掩码表示
        mask_indices = []
        for i in range(E):
            for j in range(E):
                src_block = block_ids[edge_index[0, i]]
                dst_block = block_ids[edge_index[1, j]]
                if src_block == dst_block:
                    mask_indices.append([i, j])
        
        if len(mask_indices) > 0:
            mask_indices = torch.tensor(mask_indices, dtype=torch.long, device=device).t()
            mask = torch.sparse_coo_tensor(
                mask_indices, 
                torch.ones(mask_indices.shape[1], device=device),
                (E, E)
            ).to_dense().bool()
        else:
            mask = torch.zeros(E, E, dtype=torch.bool, device=device)
    else:
        # 使用密集掩码
        mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        for i in range(E):
            for j in range(E):
                src_block = block_ids[edge_index[0, i]]
                dst_block = block_ids[edge_index[1, j]]
                if src_block == dst_block:
                    mask[i, j] = True
    
    return mask


def create_inter_block_mask(
    edge_index: torch.Tensor, 
    block_ids: torch.Tensor, 
    device: torch.device,
    use_sparse: bool = False
) -> torch.Tensor:
    """创建区块间掩码 - 优化版本"""
    E = edge_index.shape[1]
    
    if use_sparse:
        # 使用稀疏掩码表示
        mask_indices = []
        for i in range(E):
            for j in range(E):
                src_block = block_ids[edge_index[0, i]]
                dst_block = block_ids[edge_index[1, j]]
                if src_block != dst_block:
                    mask_indices.append([i, j])
        
        if len(mask_indices) > 0:
            mask_indices = torch.tensor(mask_indices, dtype=torch.long, device=device).t()
            mask = torch.sparse_coo_tensor(
                mask_indices, 
                torch.ones(mask_indices.shape[1], device=device),
                (E, E)
            ).to_dense().bool()
        else:
            mask = torch.zeros(E, E, dtype=torch.bool, device=device)
    else:
        # 使用密集掩码
        mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        for i in range(E):
            for j in range(E):
                src_block = block_ids[edge_index[0, i]]
                dst_block = block_ids[edge_index[1, j]]
                if src_block != dst_block:
                    mask[i, j] = True
    
    return mask


if __name__ == "__main__":
    # 测试优化的边注意力
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    E = 100
    N = 50
    hidden_dim = 128
    
    edge_features = torch.randn(E, hidden_dim, device=device)
    edge_coords = torch.randn(E, 3, device=device)
    edge_index = torch.randint(0, N, (2, E), device=device)
    node_coords = torch.randn(N, 3, device=device)
    block_ids = torch.randint(0, 10, (N,), device=device)
    
    # 创建掩码
    intra_mask = create_intra_block_mask(edge_index, block_ids, device)
    inter_mask = create_inter_block_mask(edge_index, block_ids, device)
    
    # 测试不同的注意力机制
    for attention_type in ['sparse', 'block', 'full']:
        print(f"\n测试 {attention_type} 注意力...")
        
        try:
            block = OptimizedESA3DBlock(
                hidden_dim=hidden_dim,
                num_heads=8,
                num_radial=64,
                cutoff=10.0,
                attention_type=attention_type,
                k_neighbors=32,
                block_size=64,
            ).to(device)
            
            # 前向传播
            with torch.no_grad():
                output_features, output_coords = block(
                    edge_features, edge_coords, edge_index, node_coords, intra_mask, inter_mask
                )
            
            print(f"  输入特征形状: {edge_features.shape}")
            print(f"  输出特征形状: {output_features.shape}")
            print(f"  输入坐标形状: {edge_coords.shape}")
            print(f"  输出坐标形状: {output_coords.shape}")
            print(f"  测试通过")
        
        except Exception as e:
            print(f"  错误: {e}")
