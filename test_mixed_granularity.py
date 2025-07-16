#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
测试混合级ESA-3D模型
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.mixed_granularity_esa3d import MixedGranularityESA3D
from data.multi_granularity_dataset import MultiGranularityDataset


def test_mixed_granularity_esa3d():
    """测试混合级ESA-3D模型"""
    print("测试混合级ESA-3D模型...")
    
    # 创建数据集
    dataset = MultiGranularityDataset(
        data_dir='./data/processed',
        split='train',
        granularity='mixed',
        max_atoms=50,  # 减少原子数用于测试
        max_residues=20,
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("数据集为空！")
        return
    
    # 获取样本
    sample = dataset[0]
    print(f"样本节点数: {sample.x.shape[0]}")
    print(f"样本边数: {sample.edge_index.shape[1]}")
    print(f"样本block数: {torch.unique(sample.block_ids).shape[0]}")
    print(f"节点特征维度: {sample.x.shape[1]}")
    print(f"边特征维度: {sample.edge_attr.shape[1]}")
    
    # 创建模型
    model = MixedGranularityESA3D(
        node_dim=sample.x.shape[1],  # 使用实际的节点特征维度
        edge_dim=sample.edge_attr.shape[1],  # 使用实际的边特征维度
        hidden_dim=64,  # 减少hidden_dim用于测试
        num_layers=2,   # 减少层数用于测试
        chunk_size=32,  # 小的chunk size
        use_sparse=True,
        top_k=16,
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        try:
            output = model(sample)
            print(f"模型输出形状: {output.shape}")
            print(f"输出值: {output.item():.4f}")
            print("混合级ESA-3D模型测试成功！")
            return True
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_batch_processing():
    """测试批处理"""
    print("\n测试批处理...")
    
    from torch_geometric.data import Batch
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return Batch.from_data_list(batch)
    
    # 创建数据集
    dataset = MultiGranularityDataset(
        data_dir='./data/processed',
        split='train',
        granularity='mixed',
        max_atoms=30,  # 进一步减少原子数
        max_residues=15,
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # 创建模型
    model = MixedGranularityESA3D(
        node_dim=43,
        edge_dim=16,
        hidden_dim=64,
        num_layers=2,
        chunk_size=32,
        use_sparse=True,
        top_k=16,
    )
    
    # 测试批处理
    for batch in dataloader:
        try:
            print(f"批次节点数: {batch.x.shape[0]}")
            print(f"批次边数: {batch.edge_index.shape[1]}")
            print(f"批次大小: {batch.batch.max().item() + 1}")
            
            model.eval()
            with torch.no_grad():
                output = model(batch)
                print(f"批次输出形状: {output.shape}")
                print("批处理测试成功！")
                return True
        except Exception as e:
            print(f"批处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # 测试单个样本
    success1 = test_mixed_granularity_esa3d()
    
    # 测试批处理
    if success1:
        success2 = test_batch_processing()
    
    print("\n所有测试完成！")
