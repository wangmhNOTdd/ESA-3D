#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
多粒度训练脚本
支持原子级、残基级、混合级建图的ESA-3D模型训练
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from data.multi_granularity_dataset import MultiGranularityDataset, create_multi_granularity_config
from models.enhanced_esa3d import EnhancedESA3DModel, create_enhanced_model_config
from utils.local_logger import LocalLogger


def collate_fn(batch):
    """自定义批处理函数"""
    return Batch.from_data_list(batch)


def train_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        # 前向传播
        output = model(batch)
        target = batch.y.view(-1, 1)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
        
        # 记录日志
        if batch_idx % 10 == 0:
            logger.log_scalar(f'train/batch_loss', loss.item(), epoch * len(loader) + batch_idx)
    
    return total_loss / total_samples


def evaluate(model, loader, criterion, device, logger, epoch, split='valid'):
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # 前向传播
            output = model(batch)
            target = batch.y.view(-1, 1)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 累积损失
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            
            # 收集预测和真实值
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())
    
    # 计算指标
    avg_loss = total_loss / total_samples
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # 记录日志
    logger.log_scalar(f'{split}/loss', avg_loss, epoch)
    logger.log_scalar(f'{split}/mae', mae, epoch)
    logger.log_scalar(f'{split}/rmse', rmse, epoch)
    
    return avg_loss, mae, rmse, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='多粒度ESA-3D训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--granularity', type=str, default='mixed', 
                       choices=['atom', 'residue', 'mixed'], help='建图粒度')
    parser.add_argument('--attention_type', type=str, default='sparse',
                       choices=['sparse', 'block', 'full'], help='注意力类型')
    parser.add_argument('--model_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'], help='模型大小')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 更新配置
    config['data']['granularity'] = args.granularity
    config['exp_name'] = f"esa3d_{args.granularity}_{args.attention_type}_{args.model_size}"
    config['save_dir'] = f"./experiments/{config['exp_name']}"
    
    # 创建模型配置
    model_config = create_enhanced_model_config(
        granularity=args.granularity,
        attention_type=args.attention_type,
        model_size=args.model_size
    )\n    \n    # 更新配置\n    config['model'].update(model_config)\n    \n    # 创建保存目录\n    os.makedirs(config['save_dir'], exist_ok=True)\n    \n    # 保存配置\n    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:\n        json.dump(config, f, indent=2)\n    \n    # 设置设备\n    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')\n    print(f\"使用设备: {device}\")\n    \n    # 设置随机种子\n    torch.manual_seed(config['random_seed'])\n    np.random.seed(config['random_seed'])\n    \n    # 创建数据集\n    print(f\"创建数据集 - 粒度: {args.granularity}\")\n    \n    train_dataset = MultiGranularityDataset(\n        data_dir=config['data_dir'],\n        split='train',\n        granularity=args.granularity,\n        max_atoms=config['data']['max_atoms'],\n        max_residues=config['data'].get('max_residues', 200),\n        atom_cutoff=config['data']['atom_cutoff'],\n        residue_cutoff=config['data'].get('residue_cutoff', 10.0),\n        include_hydrogen=config['data']['include_hydrogen'],\n    )\n    \n    valid_dataset = MultiGranularityDataset(\n        data_dir=config['data_dir'],\n        split='valid',\n        granularity=args.granularity,\n        max_atoms=config['data']['max_atoms'],\n        max_residues=config['data'].get('max_residues', 200),\n        atom_cutoff=config['data']['atom_cutoff'],\n        residue_cutoff=config['data'].get('residue_cutoff', 10.0),\n        include_hydrogen=config['data']['include_hydrogen'],\n    )\n    \n    test_dataset = MultiGranularityDataset(\n        data_dir=config['data_dir'],\n        split='test',\n        granularity=args.granularity,\n        max_atoms=config['data']['max_atoms'],\n        max_residues=config['data'].get('max_residues', 200),\n        atom_cutoff=config['data']['atom_cutoff'],\n        residue_cutoff=config['data'].get('residue_cutoff', 10.0),\n        include_hydrogen=config['data']['include_hydrogen'],\n    )\n    \n    print(f\"训练集大小: {len(train_dataset)}\")\n    print(f\"验证集大小: {len(valid_dataset)}\")\n    print(f\"测试集大小: {len(test_dataset)}\")\n    \n    # 创建数据加载器\n    train_loader = GeometricDataLoader(\n        train_dataset,\n        batch_size=config['training']['batch_size'],\n        shuffle=True,\n        num_workers=config['training']['num_workers'],\n        pin_memory=True,\n    )\n    \n    valid_loader = GeometricDataLoader(\n        valid_dataset,\n        batch_size=config['training']['batch_size'],\n        shuffle=False,\n        num_workers=config['training']['num_workers'],\n        pin_memory=True,\n    )\n    \n    test_loader = GeometricDataLoader(\n        test_dataset,\n        batch_size=config['training']['batch_size'],\n        shuffle=False,\n        num_workers=config['training']['num_workers'],\n        pin_memory=True,\n    )\n    \n    # 创建模型\n    print(f\"创建模型 - 注意力类型: {args.attention_type}, 大小: {args.model_size}\")\n    \n    model = EnhancedESA3DModel(\n        node_dim=config['model']['node_dim'],\n        edge_dim=config['model']['edge_dim'],\n        hidden_dim=config['model']['hidden_dim'],\n        num_layers=config['model']['num_layers'],\n        num_heads=config['model']['num_heads'],\n        num_radial=config['model']['num_radial'],\n        cutoff=config['model']['cutoff'],\n        num_seeds=config['model']['num_seeds'],\n        output_dim=config['model']['output_dim'],\n        granularity=args.granularity,\n        attention_type=args.attention_type,\n        k_neighbors=config['model']['k_neighbors'],\n        block_size=config['model']['block_size'],\n        dropout=config['model']['dropout'],\n    ).to(device)\n    \n    # 打印模型信息\n    total_params = sum(p.numel() for p in model.parameters())\n    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    print(f\"总参数数量: {total_params:,}\")\n    print(f\"可训练参数数量: {trainable_params:,}\")\n    \n    # 创建优化器和损失函数\n    optimizer = optim.Adam(\n        model.parameters(),\n        lr=config['training']['learning_rate'],\n        weight_decay=config['training']['weight_decay'],\n    )\n    \n    criterion = nn.MSELoss()\n    \n    # 学习率调度器\n    scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n        optimizer,\n        mode='min',\n        factor=0.5,\n        patience=config['training']['patience'] // 2,\n        verbose=True,\n    )\n    \n    # 创建日志记录器\n    logger = LocalLogger(config['save_dir'])\n    \n    # 恢复训练\n    start_epoch = 0\n    best_valid_loss = float('inf')\n    \n    if args.resume:\n        print(f\"恢复训练从: {args.resume}\")\n        checkpoint = torch.load(args.resume, map_location=device)\n        model.load_state_dict(checkpoint['model_state_dict'])\n        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        start_epoch = checkpoint['epoch'] + 1\n        best_valid_loss = checkpoint['best_valid_loss']\n    \n    # 训练循环\n    print(\"开始训练...\")\n    patience_counter = 0\n    \n    for epoch in range(start_epoch, config['training']['num_epochs']):\n        print(f\"\\nEpoch {epoch + 1}/{config['training']['num_epochs']}\")\n        \n        # 训练\n        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)\n        \n        # 验证\n        valid_loss, valid_mae, valid_rmse, _, _ = evaluate(\n            model, valid_loader, criterion, device, logger, epoch, 'valid'\n        )\n        \n        # 学习率调度\n        scheduler.step(valid_loss)\n        \n        # 记录日志\n        logger.log_scalar('train/loss', train_loss, epoch)\n        logger.log_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)\n        \n        print(f\"Train Loss: {train_loss:.4f}\")\n        print(f\"Valid Loss: {valid_loss:.4f}, MAE: {valid_mae:.4f}, RMSE: {valid_rmse:.4f}\")\n        \n        # 保存最佳模型\n        if valid_loss < best_valid_loss:\n            best_valid_loss = valid_loss\n            patience_counter = 0\n            \n            # 保存检查点\n            checkpoint = {\n                'epoch': epoch,\n                'model_state_dict': model.state_dict(),\n                'optimizer_state_dict': optimizer.state_dict(),\n                'best_valid_loss': best_valid_loss,\n                'config': config,\n            }\n            \n            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model.pt'))\n            print(f\"保存最佳模型 (Valid Loss: {best_valid_loss:.4f})\")\n        else:\n            patience_counter += 1\n        \n        # 早停\n        if patience_counter >= config['training']['patience']:\n            print(f\"早停: 验证损失在 {config['training']['patience']} 个epoch内没有改善\")\n            break\n        \n        # 定期保存检查点\n        if (epoch + 1) % 10 == 0:\n            checkpoint = {\n                'epoch': epoch,\n                'model_state_dict': model.state_dict(),\n                'optimizer_state_dict': optimizer.state_dict(),\n                'best_valid_loss': best_valid_loss,\n                'config': config,\n            }\n            torch.save(checkpoint, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pt'))\n    \n    # 最终测试\n    print(\"\\n最终测试...\")\n    \n    # 加载最佳模型\n    best_checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pt'), map_location=device)\n    model.load_state_dict(best_checkpoint['model_state_dict'])\n    \n    # 测试\n    test_loss, test_mae, test_rmse, test_predictions, test_targets = evaluate(\n        model, test_loader, criterion, device, logger, -1, 'test'\n    )\n    \n    print(f\"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}\")\n    \n    # 生成预测结果图\n    plt.figure(figsize=(8, 6))\n    plt.scatter(test_targets, test_predictions, alpha=0.7)\n    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--', lw=2)\n    plt.xlabel('True Values')\n    plt.ylabel('Predictions')\n    plt.title(f'Test Results - {args.granularity} + {args.attention_type}')\n    plt.grid(True, alpha=0.3)\n    plt.tight_layout()\n    plt.savefig(os.path.join(config['save_dir'], 'test_predictions.png'), dpi=300)\n    plt.close()\n    \n    # 保存测试结果\n    test_results = {\n        'config': config,\n        'test_loss': test_loss,\n        'test_mae': test_mae,\n        'test_rmse': test_rmse,\n        'predictions': test_predictions,\n        'targets': test_targets,\n    }\n    \n    with open(os.path.join(config['save_dir'], 'test_results.json'), 'w') as f:\n        json.dump({\n            'config': config,\n            'test_loss': test_loss,\n            'test_mae': test_mae,\n            'test_rmse': test_rmse,\n        }, f, indent=2)\n    \n    # 生成训练曲线\n    logger.plot_training_curves()\n    \n    # 生成实验报告\n    logger.generate_experiment_report({\n        'model_info': {\n            'granularity': args.granularity,\n            'attention_type': args.attention_type,\n            'model_size': args.model_size,\n            'total_params': total_params,\n            'trainable_params': trainable_params,\n        },\n        'dataset_info': {\n            'train_size': len(train_dataset),\n            'valid_size': len(valid_dataset),\n            'test_size': len(test_dataset),\n        },\n        'final_results': {\n            'test_loss': test_loss,\n            'test_mae': test_mae,\n            'test_rmse': test_rmse,\n        },\n    })\n    \n    print(f\"\\n训练完成! 结果保存在: {config['save_dir']}\")\n\n\nif __name__ == \"__main__\":\n    main()
