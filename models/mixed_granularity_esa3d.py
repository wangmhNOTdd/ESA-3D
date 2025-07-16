#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
混合级ESA-3D模型
针对混合级建图优化，使用内存高效的边注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_sum
from typing import Optional, Tuple, List

from modules.memory_efficient_attention import MemoryEfficientESA3DBlock


class MixedGranularityESA3D(nn.Module):
    """混合粒度ESA-3D模型"""
    
    def __init__(
        self,
        node_dim: int = 43,
        edge_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        num_radial: int = 64,
        cutoff: float = 10.0,
        num_seeds: int = 32,
        output_dim: int = 1,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
        chunk_size: int = 64,
        use_sparse: bool = True,
        top_k: int = 32,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.use_sparse = use_sparse
        self.top_k = top_k
        
        # 节点嵌入
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 边嵌入
        self.edge_embedding = EdgeEmbedding(
            node_dim=hidden_dim,  # 使用嵌入后的维度
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        
        # 多层ESA-3D块
        self.layers = nn.ModuleList([
            MemoryEfficientESA3DBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_radial=num_radial,
                cutoff=cutoff,
                activation=activation,
                dropout=dropout,
                chunk_size=chunk_size,
                use_sparse=use_sparse,
                top_k=top_k,
            ) for _ in range(num_layers)
        ])
        
        # 注意力池化
        self.pooling = AttentionPooling(
            hidden_dim=hidden_dim,
            num_seeds=num_seeds,
            num_heads=num_heads,
            activation=activation,
        )
        
        # 输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data对象，包含：
                - x: [N, node_dim] 节点特征
                - pos: [N, 3] 节点坐标
                - edge_index: [2, E] 边索引
                - edge_attr: [E, edge_dim] 边特征
                - block_ids: [N] 每个节点的block ID
                - batch: [N] 批次信息（可选）
        """
        x, pos, edge_index, edge_attr = data.x, data.pos, data.edge_index, data.edge_attr
        block_ids = data.block_ids
        batch = getattr(data, 'batch', None)
        
        # 节点嵌入
        node_features = self.node_embedding(x)  # [N, hidden_dim]
        
        # 边嵌入
        edge_features, edge_coords = self.edge_embedding(
            node_features, pos, edge_index, edge_attr
        )
        
        # 创建掩码
        intra_mask, inter_mask = self._create_block_masks(
            edge_index, block_ids, edge_features.device
        )
        
        # 逐层处理
        current_edge_features = edge_features
        current_edge_coords = edge_coords
        
        for layer in self.layers:
            current_edge_features, current_edge_coords = layer(
                current_edge_features,
                current_edge_coords,
                intra_mask,
                inter_mask,
                block_ids,
                edge_index,
            )
        
        # 获取批次信息
        if batch is not None:
            # 根据边索引获取边的批次信息
            edge_batch = batch[edge_index[0]]
        else:
            edge_batch = None
        
        # 注意力池化
        graph_repr = self.pooling(
            current_edge_features, current_edge_coords, edge_batch
        )
        
        # 输出预测
        output = self.output_mlp(graph_repr)
        
        return output
    
    def _create_block_masks(
        self,
        edge_index: torch.Tensor,
        block_ids: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建区块内和区块间掩码"""
        E = edge_index.shape[1]
        
        # 获取每条边对应的节点的block_ids
        src_block_ids = block_ids[edge_index[0]]  # [E]
        dst_block_ids = block_ids[edge_index[1]]  # [E]
        
        # 创建区块内掩码：两个节点在同一个block
        intra_mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        inter_mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        
        # 向量化计算：判断边是否在同一block内
        # 边i在block内当且仅当src_block_ids[i] == dst_block_ids[i]
        edge_is_intra = (src_block_ids == dst_block_ids)  # [E]
        
        # 对于每对边(i,j)，检查它们是否在同一个block
        # 使用广播来避免双重循环
        src_same = src_block_ids.unsqueeze(1) == src_block_ids.unsqueeze(0)  # [E, E]
        dst_same = dst_block_ids.unsqueeze(1) == dst_block_ids.unsqueeze(0)  # [E, E]
        
        # 区块内掩码：两条边都在同一个block内
        both_intra = edge_is_intra.unsqueeze(1) & edge_is_intra.unsqueeze(0)  # [E, E]
        same_block = src_same & dst_same  # [E, E]
        intra_mask = both_intra & same_block
        
        # 区块间掩码：至少一条边是跨block的
        at_least_one_inter = (~edge_is_intra.unsqueeze(1)) | (~edge_is_intra.unsqueeze(0))  # [E, E]
        inter_mask = at_least_one_inter
        
        return intra_mask, inter_mask
    
    def _create_optimized_block_masks(
        self,
        edge_index: torch.Tensor,
        block_ids: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建优化的区块掩码（稀疏版本）"""
        E = edge_index.shape[1]
        
        # 获取每条边对应的节点的block_ids
        src_block_ids = block_ids[edge_index[0]]  # [E]
        dst_block_ids = block_ids[edge_index[1]]  # [E]
        
        # 创建稀疏掩码列表
        intra_pairs = []
        inter_pairs = []
        
        for i in range(E):
            for j in range(E):
                if i == j:
                    continue
                
                src_i, dst_i = src_block_ids[i], dst_block_ids[i]
                src_j, dst_j = src_block_ids[j], dst_block_ids[j]
                
                # 区块内：两条边都在同一个block内
                if src_i == dst_i and src_j == dst_j and src_i == src_j:
                    intra_pairs.append([i, j])
                # 区块间：两条边连接不同的block
                elif src_i != dst_i or src_j != dst_j:
                    # 只考虑相邻的block
                    if (src_i == src_j or src_i == dst_j or 
                        dst_i == src_j or dst_i == dst_j):
                        inter_pairs.append([i, j])
        
        # 转换为稀疏掩码
        intra_mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        inter_mask = torch.zeros(E, E, dtype=torch.bool, device=device)
        
        if intra_pairs:
            intra_indices = torch.tensor(intra_pairs, device=device).t()
            intra_mask[intra_indices[0], intra_indices[1]] = True
        
        if inter_pairs:
            inter_indices = torch.tensor(inter_pairs, device=device).t()
            inter_mask[inter_indices[0], inter_indices[1]] = True
        
        return intra_mask, inter_mask


class EdgeEmbedding(nn.Module):
    """边嵌入层"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 节点特征到边特征的映射
        input_dim = node_dim * 2
        if edge_dim > 0:
            input_dim += edge_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            activation,
            nn.Linear(hidden_dim * 2, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 边的坐标特征：相对位置向量
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,  # [N, node_dim]
        node_coords: torch.Tensor,    # [N, 3]
        edge_index: torch.Tensor,     # [2, E]
        edge_attr: Optional[torch.Tensor] = None,  # [E, edge_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [N, node_dim] 节点特征
            node_coords: [N, 3] 节点坐标
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征（可选）
        
        Returns:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        
        # 获取边的节点特征
        src_features = node_features[src_idx]  # [E, node_dim]
        dst_features = node_features[dst_idx]  # [E, node_dim]
        
        # 拼接节点特征
        edge_node_features = torch.cat([src_features, dst_features], dim=-1)  # [E, 2*node_dim]
        
        # 如果有边特征，也拼接上
        if edge_attr is not None:
            edge_node_features = torch.cat([edge_node_features, edge_attr], dim=-1)
        
        # 生成边的不变特征
        edge_features = self.edge_mlp(edge_node_features)  # [E, hidden_dim]
        
        # 生成边的等变特征（相对位置向量）
        src_coords = node_coords[src_idx]  # [E, 3]
        dst_coords = node_coords[dst_idx]  # [E, 3]
        relative_coords = dst_coords - src_coords  # [E, 3]
        
        edge_coords = self.coord_mlp(relative_coords)  # [E, 3]
        
        return edge_features, edge_coords


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int = 32,
        num_heads: int = 8,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # 可学习的种子向量
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, hidden_dim))
        
        # 多头注意力
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 最终投影
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_seeds, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        edge_features: torch.Tensor,  # [E, hidden_dim]
        edge_coords: torch.Tensor,    # [E, 3]
        batch: Optional[torch.Tensor] = None,  # [E] 批次信息
    ) -> torch.Tensor:
        """
        Args:
            edge_features: [E, hidden_dim] 边的不变特征
            edge_coords: [E, 3] 边的等变特征
            batch: [E] 批次信息
        
        Returns:
            graph_repr: [B, hidden_dim] 图的全局表示
        """
        if batch is None:
            batch = torch.zeros(edge_features.shape[0], dtype=torch.long, device=edge_features.device)
        
        batch_size = int(batch.max().item() + 1)
        
        # 为每个批次处理
        graph_reprs = []
        
        for b in range(batch_size):
            mask = batch == b
            if mask.sum() == 0:
                continue
            
            edge_feat_b = edge_features[mask]  # [E_b, hidden_dim]
            E_b = edge_feat_b.shape[0]
            
            # 种子向量作为query
            seeds = self.seed_vectors.unsqueeze(0).expand(1, -1, -1)  # [1, num_seeds, hidden_dim]
            
            # 投影
            q = self.q_proj(seeds).view(1, self.num_seeds, self.num_heads, self.head_dim)  # [1, num_seeds, num_heads, head_dim]
            k = self.k_proj(edge_feat_b).view(E_b, self.num_heads, self.head_dim)  # [E_b, num_heads, head_dim]
            v = self.v_proj(edge_feat_b).view(E_b, self.num_heads, self.head_dim)  # [E_b, num_heads, head_dim]
            
            # 多头注意力
            attended_features = []
            for head in range(self.num_heads):
                q_head = q[0, :, head, :]  # [num_seeds, head_dim]
                k_head = k[:, head, :]  # [E_b, head_dim]
                v_head = v[:, head, :]  # [E_b, head_dim]
                
                # 注意力分数
                attention_scores = torch.matmul(q_head, k_head.t())  # [num_seeds, E_b]
                attention_weights = torch.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
                
                # 加权聚合
                attended_feat = torch.matmul(attention_weights, v_head)  # [num_seeds, head_dim]
                attended_features.append(attended_feat)
            
            # 拼接多头结果
            attended_features = torch.cat(attended_features, dim=-1)  # [num_seeds, hidden_dim]
            
            # 应用输出投影和残差连接
            attended_features = self.norm(seeds[0] + self.out_proj(attended_features))
            
            # 展平并通过最终MLP
            graph_repr = self.final_mlp(attended_features.view(-1))  # [hidden_dim]
            graph_reprs.append(graph_repr)
        
        return torch.stack(graph_reprs, dim=0)  # [B, hidden_dim]


# 创建混合级配置
def create_mixed_granularity_config():
    """创建混合级建图的配置"""
    return {
        "exp_name": "esa3d_mixed_granularity",
        "data_dir": "./data/processed",
        "save_dir": "./experiments/mixed_granularity",
        "device": "cuda",
        "random_seed": 42,
        
        "model": {
            "node_dim": 43,  # element(11) + residue(21) + atom_pos(11)
            "edge_dim": 16,
            "hidden_dim": 128,
            "num_layers": 6,
            "num_heads": 8,
            "num_radial": 64,
            "cutoff": 10.0,
            "num_seeds": 32,
            "output_dim": 1,
            "dropout": 0.1,
            "chunk_size": 64,  # 内存优化参数
            "use_sparse": True,
            "top_k": 32,
        },
        
        "data": {
            "granularity": "mixed",
            "max_atoms": 1000,
            "max_residues": 200,
            "atom_cutoff": 5.0,
            "residue_cutoff": 10.0,
            "include_hydrogen": False
        },
        
        "training": {
            "batch_size": 4,
            "num_epochs": 200,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "patience": 30,
            "num_workers": 4
        }
    }


if __name__ == "__main__":
    # 测试混合级模型
    import torch
    from torch_geometric.data import Data, Batch
    
    # 创建测试数据
    N = 100  # 节点数
    E = 300  # 边数
    
    # 模拟数据
    x = torch.randn(N, 43)  # 节点特征
    pos = torch.randn(N, 3)  # 节点坐标
    edge_index = torch.randint(0, N, (2, E))  # 边索引
    edge_attr = torch.randn(E, 16)  # 边特征
    block_ids = torch.randint(0, 20, (N,))  # block ID
    y = torch.randn(1)  # 标签
    
    # 创建PyG数据
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, 
                block_ids=block_ids, y=y)
    
    # 创建模型
    model = MixedGranularityESA3D(
        node_dim=43,
        edge_dim=16,
        hidden_dim=128,
        num_layers=3,  # 减少层数用于测试
        chunk_size=32,  # 小的chunk size用于测试
        use_sparse=True,
        top_k=16,
    )
    
    # 前向传播
    try:
        output = model(data)
        print(f"模型输出形状: {output.shape}")
        print(f"输出值: {output}")
        print("混合级ESA-3D模型测试成功!")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
