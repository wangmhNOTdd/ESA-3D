#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
混合级ESA-3D训练脚本
支持内存优化和多粒度建图
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 导入模型和数据
from models.mixed_granularity_esa3d import MixedGranularityESA3D, create_mixed_granularity_config
from data.multi_granularity_dataset import MultiGranularityDataset
from utils.local_logger import LocalLogger


def collate_fn(batch):
    """自定义批处理函数"""
    return Batch.from_data_list(batch)


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        try:
            batch = batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(batch)
            
            # 计算损失
            loss = criterion(output.squeeze(), batch.y.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
        except Exception as e:
            print(f"训练批次出错: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            try:
                batch = batch.to(device)
                
                # 前向传播
                output = model(batch)
                
                # 计算损失
                loss = criterion(output.squeeze(), batch.y.squeeze())
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(batch.y.squeeze().cpu().numpy())
                
            except Exception as e:
                print(f"评估批次出错: {e}")
                continue
    
    # 计算指标
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'targets': targets,
    }


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="混合级ESA-3D训练")
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./data/processed', help='数据目录')
    parser.add_argument('--save_dir', type=str, default='./experiments/mixed_granularity', help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--chunk_size', type=int, default=64, help='内存优化chunk大小')
    parser.add_argument('--use_sparse', action='store_true', help='使用稀疏注意力')
    parser.add_argument('--top_k', type=int, default=32, help='稀疏注意力的top-k')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_mixed_granularity_config()
    
    # 更新配置
    config['data']['data_dir'] = args.data_dir
    config['save_dir'] = args.save_dir
    config['device'] = args.device
    config['training']['batch_size'] = args.batch_size
    config['training']['num_epochs'] = args.num_epochs
    config['training']['learning_rate'] = args.learning_rate
    config['model']['chunk_size'] = args.chunk_size
    config['model']['use_sparse'] = args.use_sparse
    config['model']['top_k'] = args.top_k
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("加载数据集...")
    try:
        train_dataset = MultiGranularityDataset(
            data_dir=args.data_dir,
            split='train',
            granularity='mixed',
            max_atoms=config['data']['max_atoms'],
            max_residues=config['data']['max_residues'],
            atom_cutoff=config['data']['atom_cutoff'],
            residue_cutoff=config['data']['residue_cutoff'],
            include_hydrogen=config['data']['include_hydrogen'],
        )
        
        valid_dataset = MultiGranularityDataset(
            data_dir=args.data_dir,
            split='valid',
            granularity='mixed',
            max_atoms=config['data']['max_atoms'],
            max_residues=config['data']['max_residues'],
            atom_cutoff=config['data']['atom_cutoff'],
            residue_cutoff=config['data']['residue_cutoff'],
            include_hydrogen=config['data']['include_hydrogen'],
        )
        
        test_dataset = MultiGranularityDataset(
            data_dir=args.data_dir,
            split='test',
            granularity='mixed',
            max_atoms=config['data']['max_atoms'],
            max_residues=config['data']['max_residues'],
            atom_cutoff=config['data']['atom_cutoff'],
            residue_cutoff=config['data']['residue_cutoff'],
            include_hydrogen=config['data']['include_hydrogen'],
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(valid_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=config['training']['num_workers'],
            collate_fn=collate_fn,
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=collate_fn,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=collate_fn,
        )
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建模型
    print("创建模型...")
    model = MixedGranularityESA3D(
        node_dim=config['model']['node_dim'],
        edge_dim=config['model']['edge_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_radial=config['model']['num_radial'],
        cutoff=config['model']['cutoff'],
        num_seeds=config['model']['num_seeds'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout'],
        chunk_size=config['model']['chunk_size'],
        use_sparse=config['model']['use_sparse'],
        top_k=config['model']['top_k'],
    ).to(device)
    
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 创建日志记录器
    logger = LocalLogger(args.save_dir)
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config['training']['grad_clip']
        )
        
        # 验证
        val_results = evaluate(model, valid_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_results['loss'])
        
        # 记录日志
        logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_results['loss'],
            'val_mse': val_results['mse'],
            'val_mae': val_results['mae'],
            'val_rmse': val_results['rmse'],
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_results['loss']:.4f}")
        print(f"验证MSE: {val_results['mse']:.4f}")
        print(f"验证MAE: {val_results['mae']:.4f}")
        print(f"验证RMSE: {val_results['rmse']:.4f}")
        
        # 保存最佳模型
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': val_results['loss'],
            }, os.path.join(args.save_dir, 'best_model.pth'))
            
            print(f"保存最佳模型 (验证损失: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['training']['patience']:
            print(f"早停在epoch {epoch + 1}")
            break
    
    # 最终测试
    print("\\n最终测试...")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试
    test_results = evaluate(model, test_loader, criterion, device)
    
    print(f"测试损失: {test_results['loss']:.4f}")
    print(f"测试MSE: {test_results['mse']:.4f}")
    print(f"测试MAE: {test_results['mae']:.4f}")
    print(f"测试RMSE: {test_results['rmse']:.4f}")
    
    # 保存测试结果
    test_results_file = os.path.join(args.save_dir, 'test_results.json')
    with open(test_results_file, 'w') as f:
        json.dump({
            'test_loss': test_results['loss'],
            'test_mse': test_results['mse'],
            'test_mae': test_results['mae'],
            'test_rmse': test_results['rmse'],
        }, f, indent=2)
    
    # 生成训练曲线
    logger.plot_training_curves()
    
    # 生成实验报告
    logger.generate_experiment_report({
        'model_name': 'Mixed Granularity ESA-3D',
        'dataset': 'PDBBind',
        'final_test_results': {
            'loss': test_results['loss'],
            'mse': test_results['mse'],
            'mae': test_results['mae'],
            'rmse': test_results['rmse'],
        },
        'model_parameters': count_parameters(model),
        'training_time': time.time(),
    })
    
    print(f"训练完成！结果保存在: {args.save_dir}")


if __name__ == "__main__":
    main()
