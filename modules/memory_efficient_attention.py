#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
内存高效的边注意力实现
使用分块处理来解决O(E²)内存瓶颈
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum
from typing import Optional, Tuple, List
import numpy as np

from .equivariant_edge_attention import RadialBasisFunction


class MemoryEfficientEdgeAttention(nn.Module):
    """内存高效的边注意力模块"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
        chunk_size: int = 64,  # 分块大小
        use_sparse: bool = True,  # 使用稀疏注意力
        top_k: int = 32,  # 每个边最多关注的邻居数
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff = cutoff
        self.activation = activation
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.use_sparse = use_sparse
        self.top_k = top_k
        
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
        
        self._init_weights()
    
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        edge_mask: Optional[torch.Tensor] = None,  # [E, E] 或 None
        block_ids: Optional[torch.Tensor] = None,  # [N] 节点的block ID
        edge_index: Optional[torch.Tensor] = None,  # [2, E] 边的索引
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            edge_mask: [E, E] 边-边连接掩码，或None使用全连接
            block_ids: [N] 节点的block ID，用于块内注意力
            edge_index: [2, E] 边的索引
        
        Returns:
            updated_features: [E, hidden_dim] 更新后的边不变特征
            updated_coords: [E, 3] 更新后的边等变特征
        """
        E = edge_features.shape[0]
        device = edge_features.device
        
        # 投影到Q, K, V
        q = self.q_proj(edge_features).view(E, self.num_heads, self.head_dim)
        k = self.k_proj(edge_features).view(E, self.num_heads, self.head_dim)
        v = self.v_proj(edge_features).view(E, self.num_heads, self.head_dim)
        
        # 如果使用稀疏注意力且没有提供edge_mask，构建KNN掩码
        if self.use_sparse and edge_mask is None:
            edge_mask = self._build_knn_mask(edge_coords, edge_index, block_ids)
        
        # 选择注意力计算方式
        if edge_mask is not None and self.use_sparse:
            # 稀疏注意力
            updated_features, updated_coords = self._sparse_attention(
                q, k, v, edge_features, edge_coords, edge_mask
            )
        elif E <= self.chunk_size:
            # 小图直接计算
            updated_features, updated_coords = self._full_attention(
                q, k, v, edge_features, edge_coords, edge_mask
            )
        else:
            # 大图使用分块注意力
            updated_features, updated_coords = self._chunked_attention(
                q, k, v, edge_features, edge_coords, edge_mask
            )
        
        # 残差连接和归一化
        updated_features = self.norm(edge_features + self.out_proj(updated_features))
        
        return updated_features, updated_coords
    
    def _build_knn_mask(
        self,
        edge_coords: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        block_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """构建KNN稀疏掩码"""
        E = edge_coords.shape[0]
        device = edge_coords.device
        
        # 计算距离矩阵
        coord_diff = edge_coords.unsqueeze(1) - edge_coords.unsqueeze(0)  # [E, E, 3]
        distances = torch.norm(coord_diff, dim=-1)  # [E, E]
        
        # 找到每个边的top-k邻居
        mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        
        # 对于每个边，找到最近的k个邻居
        _, indices = torch.topk(distances, k=min(self.top_k, E), dim=1, largest=False)
        
        # 设置掩码
        for i in range(E):
            mask[i, indices[i]] = True
        
        # 确保对称性
        mask = mask | mask.t()
        
        # 如果有block_ids，也考虑同一block内的边
        if block_ids is not None and edge_index is not None:
            # 获取每条边对应的节点的block_ids
            edge_block_ids = block_ids[edge_index[0]]  # [E]
            
            # 同一block内的边互相连接
            same_block_mask = edge_block_ids.unsqueeze(1) == edge_block_ids.unsqueeze(0)  # [E, E]
            mask = mask | same_block_mask
        
        return mask
    
    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """稀疏注意力计算"""
        E = edge_features.shape[0]
        device = edge_features.device
        
        # 获取有效的边对
        edge_pairs = torch.nonzero(edge_mask, as_tuple=False)  # [num_valid_pairs, 2]
        
        if edge_pairs.shape[0] == 0:
            return torch.zeros_like(edge_features), edge_coords
        
        # 计算相对位置和距离
        coord_diff = edge_coords[edge_pairs[:, 0]] - edge_coords[edge_pairs[:, 1]]  # [num_valid_pairs, 3]
        distances = torch.norm(coord_diff, dim=-1)  # [num_valid_pairs]
        
        # 径向基函数
        rbf_features = self.rbf(distances)  # [num_valid_pairs, num_radial]
        
        # 计算attention scores
        head_dim = self.hidden_dim // self.num_heads
        updated_features = torch.zeros(E, self.hidden_dim, device=device)
        updated_coords = edge_coords.clone()
        
        for head in range(self.num_heads):
            # 获取当前头的q, k, v
            q_head = q[:, head, :]  # [E, head_dim]
            k_head = k[:, head, :]  # [E, head_dim]
            v_head = v[:, head, :]  # [E, head_dim]
            
            # 获取有效边对的特征
            q_pairs = q_head[edge_pairs[:, 0]]  # [num_valid_pairs, head_dim]
            k_pairs = k_head[edge_pairs[:, 1]]  # [num_valid_pairs, head_dim]
            
            # 计算点积
            dot_products = torch.sum(q_pairs * k_pairs, dim=-1)  # [num_valid_pairs]
            
            # 构建attention输入
            attention_input = torch.cat([
                q_pairs,  # [num_valid_pairs, head_dim]
                k_pairs,  # [num_valid_pairs, head_dim]
                rbf_features,  # [num_valid_pairs, num_radial]
                dot_products.unsqueeze(-1),  # [num_valid_pairs, 1]
            ], dim=-1)
            
            # 计算attention scores
            attention_scores = self.attention_mlp(attention_input)[:, head]  # [num_valid_pairs]
            
            # 使用scatter_softmax进行归一化
            attention_weights = scatter_softmax(
                attention_scores, edge_pairs[:, 0], dim=0, dim_size=E
            )  # [num_valid_pairs]
            
            # 聚合特征
            v_pairs = v_head[edge_pairs[:, 1]]  # [num_valid_pairs, head_dim]
            weighted_v = attention_weights.unsqueeze(-1) * v_pairs  # [num_valid_pairs, head_dim]
            
            aggregated_v = scatter_sum(
                weighted_v, edge_pairs[:, 0], dim=0, dim_size=E
            )  # [E, head_dim]
            
            # 将聚合特征放入对应的头部位置
            updated_features[:, head * head_dim:(head + 1) * head_dim] = aggregated_v
            
            # 更新坐标
            gates = self.coord_gate_mlp(v_pairs)  # [num_valid_pairs, 1]
            weighted_coord_diff = attention_weights.unsqueeze(-1) * gates * coord_diff  # [num_valid_pairs, 3]
            
            coord_update = scatter_sum(
                weighted_coord_diff, edge_pairs[:, 0], dim=0, dim_size=E
            )  # [E, 3]
            
            updated_coords += coord_update / self.num_heads
        
        return updated_features, updated_coords
    
    def _full_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整注意力计算（用于小图）"""
        E = edge_features.shape[0]
        device = edge_features.device
        
        # 计算相对位置和距离
        coord_diff = edge_coords.unsqueeze(1) - edge_coords.unsqueeze(0)  # [E, E, 3]
        distances = torch.norm(coord_diff, dim=-1)  # [E, E]
        
        # 径向基函数
        rbf_features = self.rbf(distances)  # [E, E, num_radial]
        
        # 计算点积
        dot_products = torch.sum(
            q.unsqueeze(1) * k.unsqueeze(0), dim=-1
        )  # [E, E, num_heads]
        
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
                q_expanded,
                k_expanded,
                rbf_features,
                dot_products[:, :, head].unsqueeze(-1),
            ], dim=-1)
            
            # 计算attention scores
            attention_scores = self.attention_mlp(attention_input)[:, :, head]  # [E, E]
            
            # 应用掩码
            if edge_mask is not None:
                attention_scores = attention_scores.masked_fill(~edge_mask, float('-inf'))
            
            # Softmax归一化
            attention_weights_head = torch.softmax(attention_scores, dim=-1)  # [E, E]
            attention_weights.append(attention_weights_head)
        
        attention_weights = torch.stack(attention_weights, dim=0)  # [num_heads, E, E]
        
        # 聚合特征和坐标
        updated_features = []
        updated_coords = edge_coords.clone()
        
        for head in range(self.num_heads):
            v_head = v[:, head, :]  # [E, head_dim]
            attention_head = attention_weights[head]  # [E, E]
            
            # 聚合特征
            updated_v = torch.matmul(attention_head, v_head)  # [E, head_dim]
            updated_features.append(updated_v)
            
            # 更新坐标
            gates = self.coord_gate_mlp(v_head)  # [E, 1]
            gates_expanded = gates.unsqueeze(1).expand(-1, E, -1)  # [E, E, 1]
            
            weighted_coord_diff = (
                attention_head.unsqueeze(-1) * gates_expanded * coord_diff
            )  # [E, E, 3]
            
            coord_update = torch.sum(weighted_coord_diff, dim=1)  # [E, 3]
            updated_coords += coord_update / self.num_heads
        
        updated_features = torch.cat(updated_features, dim=-1)  # [E, hidden_dim]
        
        return updated_features, updated_coords
    
    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分块注意力计算（用于大图）"""
        E = edge_features.shape[0]
        device = edge_features.device
        
        updated_features = torch.zeros_like(edge_features)
        updated_coords = edge_coords.clone()
        
        # 分块处理
        for i in range(0, E, self.chunk_size):
            end_i = min(i + self.chunk_size, E)
            
            for j in range(0, E, self.chunk_size):
                end_j = min(j + self.chunk_size, E)
                
                # 获取块
                q_chunk = q[i:end_i]  # [chunk_size, num_heads, head_dim]
                k_chunk = k[j:end_j]  # [chunk_size, num_heads, head_dim]
                v_chunk = v[j:end_j]  # [chunk_size, num_heads, head_dim]
                
                coords_i = edge_coords[i:end_i]  # [chunk_size, 3]
                coords_j = edge_coords[j:end_j]  # [chunk_size, 3]
                
                # 计算相对位置
                coord_diff = coords_i.unsqueeze(1) - coords_j.unsqueeze(0)  # [chunk_size, chunk_size, 3]
                distances = torch.norm(coord_diff, dim=-1)  # [chunk_size, chunk_size]
                
                # 径向基函数
                rbf_features = self.rbf(distances)  # [chunk_size, chunk_size, num_radial]
                
                # 计算点积
                dot_products = torch.sum(
                    q_chunk.unsqueeze(2) * k_chunk.unsqueeze(1), dim=-1
                )  # [chunk_size, chunk_size, num_heads]
                
                # 为每个头计算注意力
                for head in range(self.num_heads):
                    q_head = q_chunk[:, head, :]  # [chunk_size, head_dim]
                    k_head = k_chunk[:, head, :]  # [chunk_size, head_dim]
                    v_head = v_chunk[:, head, :]  # [chunk_size, head_dim]
                    
                    # 构建attention输入
                    q_expanded = q_head.unsqueeze(1).expand(-1, end_j - j, -1)
                    k_expanded = k_head.unsqueeze(0).expand(end_i - i, -1, -1)
                    
                    attention_input = torch.cat([
                        q_expanded,
                        k_expanded,
                        rbf_features,
                        dot_products[:, :, head].unsqueeze(-1),
                    ], dim=-1)
                    
                    # 计算attention scores
                    attention_scores = self.attention_mlp(attention_input)[:, :, head]
                    
                    # 应用掩码
                    if edge_mask is not None:
                        mask_chunk = edge_mask[i:end_i, j:end_j]
                        attention_scores = attention_scores.masked_fill(~mask_chunk, float('-inf'))
                    
                    # Softmax归一化（沿着j维度）
                    attention_weights = torch.softmax(attention_scores, dim=-1)
                    
                    # 聚合特征
                    updated_v = torch.matmul(attention_weights, v_head)  # [chunk_size, head_dim]
                    updated_features[i:end_i] += updated_v
                    
                    # 更新坐标
                    gates = self.coord_gate_mlp(v_head)  # [chunk_size, 1]
                    gates_expanded = gates.unsqueeze(0).expand(end_i - i, -1, -1)
                    
                    weighted_coord_diff = (
                        attention_weights.unsqueeze(-1) * gates_expanded * coord_diff
                    )
                    
                    coord_update = torch.sum(weighted_coord_diff, dim=1)
                    updated_coords[i:end_i] += coord_update / self.num_heads
        
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


class MemoryEfficientESA3DBlock(nn.Module):
    """内存高效的ESA-3D块"""
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
        chunk_size: int = 64,
        use_sparse: bool = True,
        top_k: int = 32,
    ):
        super().__init__()
        
        # 区块内注意力（稀疏）
        self.intra_block_attention = MemoryEfficientEdgeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            activation=activation,
            dropout=dropout,
            chunk_size=chunk_size,
            use_sparse=True,  # 区块内使用稀疏注意力
            top_k=top_k,
        )
        
        # 区块间注意力（稀疏）
        self.inter_block_attention = MemoryEfficientEdgeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_radial=num_radial,
            cutoff=cutoff,
            activation=activation,
            dropout=dropout,
            chunk_size=chunk_size,
            use_sparse=True,  # 区块间也使用稀疏注意力
            top_k=top_k // 2,  # 区块间邻居数较少
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        edge_features: torch.Tensor,
        edge_coords: torch.Tensor,
        intra_mask: torch.Tensor,
        inter_mask: torch.Tensor,
        block_ids: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_features: [E, hidden_dim]
            edge_coords: [E, 3]
            intra_mask: [E, E] 区块内掩码
            inter_mask: [E, E] 区块间掩码
            block_ids: [N] 节点的block ID
            edge_index: [2, E] 边的索引
        """
        # 区块内注意力
        intra_features, intra_coords = self.intra_block_attention(
            edge_features, edge_coords, intra_mask, block_ids, edge_index
        )
        
        # 区块间注意力
        inter_features, inter_coords = self.inter_block_attention(
            edge_features, edge_coords, inter_mask, block_ids, edge_index
        )
        
        # 特征融合
        fused_features = self.feature_fusion(
            torch.cat([intra_features, inter_features], dim=-1)
        )
        
        # 坐标融合（简单平均）
        fused_coords = (intra_coords + inter_coords) / 2
        
        # 残差连接
        output_features = self.norm(edge_features + fused_features)
        
        return output_features, fused_coords
