#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch_geometric.data import Data, Batch
import sys
import os

# 添加路径
sys.path.append('c:/Users/18778/Desktop/torch-learn/ESA-3D')

from models.esa3d import ESA3DModel
from modules.equivariant_edge_attention import create_intra_block_mask, create_inter_block_mask


def test_esa3d_model():
    """测试ESA-3D模型的基本功能"""
    print("Testing ESA-3D model...")
    
    # 创建模型
    model = ESA3DModel(
        node_dim=41,
        edge_dim=16,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_radial=16,
        cutoff=10.0,
        num_seeds=8,
        output_dim=1,
        dropout=0.1,
    )
    
    # 创建测试数据
    batch_size = 2
    
    # 第一个分子
    num_atoms1 = 10
    x1 = torch.randn(num_atoms1, 41)
    pos1 = torch.randn(num_atoms1, 3) * 5
    block_ids1 = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    
    # 创建边（基于距离）
    distances1 = torch.norm(pos1.unsqueeze(0) - pos1.unsqueeze(1), dim=2)
    edge_indices1 = torch.where((distances1 < 5.0) & (distances1 > 0))
    edge_index1 = torch.stack(edge_indices1)
    edge_attr1 = torch.randn(edge_index1.shape[1], 16)
    
    # 第二个分子
    num_atoms2 = 8
    x2 = torch.randn(num_atoms2, 41)
    pos2 = torch.randn(num_atoms2, 3) * 5
    block_ids2 = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    
    # 创建边（基于距离）
    distances2 = torch.norm(pos2.unsqueeze(0) - pos2.unsqueeze(1), dim=2)
    edge_indices2 = torch.where((distances2 < 5.0) & (distances2 > 0))
    edge_index2 = torch.stack(edge_indices2)
    edge_attr2 = torch.randn(edge_index2.shape[1], 16)
    
    # 创建批次
    data_list = [
        Data(x=x1, pos=pos1, edge_index=edge_index1, edge_attr=edge_attr1, block_ids=block_ids1),
        Data(x=x2, pos=pos2, edge_index=edge_index2, edge_attr=edge_attr2, block_ids=block_ids2)
    ]
    
    batch = Batch.from_data_list(data_list)
    
    # 前向传播
    print(f"Input shapes:")
    print(f"  x: {batch.x.shape}")
    print(f"  pos: {batch.pos.shape}")
    print(f"  edge_index: {batch.edge_index.shape}")
    print(f"  edge_attr: {batch.edge_attr.shape}")
    print(f"  block_ids: {batch.block_ids.shape}")
    print(f"  batch: {batch.batch.shape}")
    
    try:
        output = model(
            node_features=batch.x,
            node_coords=batch.pos,
            edge_index=batch.edge_index,
            block_ids=batch.block_ids,
            batch=batch.batch,
            edge_attr=batch.edge_attr
        )
        
        print(f"\nOutput shape: {output.shape}")
        print(f"Output: {output}")
        print("\n✓ ESA-3D model test passed!")
        
    except Exception as e:
        print(f"\n✗ ESA-3D model test failed: {e}")
        import traceback
        traceback.print_exc()


def test_mask_creation():
    """测试掩码创建功能"""
    print("\nTesting mask creation...")
    
    # 创建测试数据
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    block_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    device = torch.device('cpu')
    
    # 测试区块内掩码
    try:
        intra_mask = create_intra_block_mask(edge_index, block_ids, device)
        print(f"Intra-block mask shape: {intra_mask.shape}")
        print(f"Intra-block mask:\n{intra_mask}")
        print("✓ Intra-block mask test passed!")
    except Exception as e:
        print(f"✗ Intra-block mask test failed: {e}")
    
    # 测试区块间掩码
    try:
        inter_mask = create_inter_block_mask(edge_index, block_ids, device)
        print(f"\nInter-block mask shape: {inter_mask.shape}")
        print(f"Inter-block mask:\n{inter_mask}")
        print("✓ Inter-block mask test passed!")
    except Exception as e:
        print(f"✗ Inter-block mask test failed: {e}")


def test_equivariance():
    """测试等变性"""
    print("\nTesting equivariance...")
    
    # 创建模型
    model = ESA3DModel(
        node_dim=41,
        edge_dim=16,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
        num_radial=16,
        cutoff=10.0,
        num_seeds=8,
        output_dim=1,
        dropout=0.0,  # 关闭dropout确保确定性
    )
    
    model.eval()
    
    # 创建测试数据
    num_atoms = 6
    x = torch.randn(num_atoms, 41)
    pos = torch.randn(num_atoms, 3) * 3
    block_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    
    # 创建边
    distances = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), dim=2)
    edge_indices = torch.where((distances < 5.0) & (distances > 0))
    edge_index = torch.stack(edge_indices)
    edge_attr = torch.randn(edge_index.shape[1], 16)
    
    # 原始预测
    with torch.no_grad():
        output1 = model(
            node_features=x,
            node_coords=pos,
            edge_index=edge_index,
            block_ids=block_ids,
            edge_attr=edge_attr
        )
    
    # 应用旋转和平移
    rotation_matrix = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
    translation = torch.tensor([1.0, 2.0, 3.0])
    
    pos_transformed = torch.matmul(pos, rotation_matrix.T) + translation
    
    # 变换后的预测
    with torch.no_grad():
        output2 = model(
            node_features=x,
            node_coords=pos_transformed,
            edge_index=edge_index,
            block_ids=block_ids,
            edge_attr=edge_attr
        )
    
    # 检查等变性
    diff = torch.abs(output1 - output2).max()
    print(f"Original output: {output1}")
    print(f"Transformed output: {output2}")
    print(f"Maximum difference: {diff}")
    
    if diff < 1e-3:
        print("✓ Equivariance test passed!")
    else:
        print("✗ Equivariance test failed!")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    test_mask_creation()
    test_esa3d_model()
    test_equivariance()
    
    print("\nAll tests completed!")
