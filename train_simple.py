#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
简化版的ESA-3D训练脚本
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
import argparse

from data.pdbbind_dataset import PDBBindESA3DDataset
from models.esa3d_simple import SimpleESA3DModel
from utils.local_logger import LocalLogger


def collate_fn(batch):
    """自定义批处理函数"""
    return Batch.from_data_list(batch)


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    # 计算皮尔逊相关系数
    if len(predictions) > 1:
        pearson_r, _ = pearsonr(predictions, targets)
    else:
        pearson_r = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson_r': pearson_r,
    }


def train_epoch(model, dataloader, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(
            x=batch.x,
            pos=batch.pos,
            edge_index=batch.edge_index,
            block_ids=batch.block_ids,
            batch=batch.batch,
            edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None
        )
        
        # 计算损失
        loss = F.mse_loss(pred.squeeze(), batch.y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 记录日志
        logger.log_metric('train_loss_batch', loss.item())
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            # 前向传播
            pred = model(
                x=batch.x,
                pos=batch.pos,
                edge_index=batch.edge_index,
                block_ids=batch.block_ids,
                batch=batch.batch,
                edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None
            )
            
            # 计算损失
            loss = F.mse_loss(pred.squeeze(), batch.y)
            total_loss += loss.item()
            
            # 收集预测和真实值
            predictions.extend(pred.squeeze().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    # 计算指标
    metrics = calculate_metrics(predictions, targets)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Simple ESA-3D model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 覆盖配置
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.device:
        config['device'] = args.device
    
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = PDBBindESA3DDataset(
        config['data_dir'], 
        'train', 
        max_atoms=config['data']['max_atoms'],
        cutoff=config['data']['cutoff'],
        include_hydrogen=config['data']['include_hydrogen']
    )
    
    val_dataset = PDBBindESA3DDataset(
        config['data_dir'], 
        'valid', 
        max_atoms=config['data']['max_atoms'],
        cutoff=config['data']['cutoff'],
        include_hydrogen=config['data']['include_hydrogen']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 避免多进程问题
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    print(f"Training set: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Validation set: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # 创建模型
    model = SimpleESA3DModel(
        node_dim=config['model']['node_dim'],
        edge_dim=config['model']['edge_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dropout'],
        cutoff=config['data']['cutoff'],
        num_radial=config['model']['num_radial'],
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    # 创建日志记录器
    logger = LocalLogger(config['save_dir'], config['exp_name'])
    logger.save_config(config)
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, logger)
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 记录日志
        logger.log_metric('train_loss', train_loss)
        logger.log_metric('val_loss', val_metrics['loss'])
        logger.log_metric('val_mae', val_metrics['mae'])
        logger.log_metric('val_rmse', val_metrics['rmse'])
        logger.log_metric('val_pearson_r', val_metrics['pearson_r'])
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
              f"RMSE: {val_metrics['rmse']:.4f}, Pearson R: {val_metrics['pearson_r']:.4f}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            
            print(f"Best model saved with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
