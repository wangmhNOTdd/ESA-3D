#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import numpy as np
from typing import Dict, List
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

# 导入我们的模块
from models.esa3d import ESA3DModel
from data.pdbbind_dataset import PDBBindESA3DDataset, PDBBindESA3DCollator
from utils.local_logger import LocalLogger


class ESA3DTrainer:
    """ESA-3D训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        # 创建模型
        self.model = ESA3DModel(
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
        ).to(self.device)
        
        # 创建数据加载器
        self.train_loader, self.valid_loader, self.test_loader = self._create_data_loaders()
        
        # 创建优化器和调度器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # 初始化本地日志记录器
        self.logger = LocalLogger(
            log_dir=os.path.join(config['save_dir'], 'logs'),
            experiment_name=config['exp_name']
        )
        
        # 保存配置
        self.logger.save_config(config)
    
    def _create_data_loaders(self) -> tuple:
        """创建数据加载器"""
        
        # 创建数据集
        train_dataset = PDBBindESA3DDataset(
            data_dir=self.config['data_dir'],
            split='train',
            max_atoms=self.config['data']['max_atoms'],
            cutoff=self.config['data']['cutoff'],
            include_hydrogen=self.config['data']['include_hydrogen']
        )
        
        valid_dataset = PDBBindESA3DDataset(
            data_dir=self.config['data_dir'],
            split='valid',
            max_atoms=self.config['data']['max_atoms'],
            cutoff=self.config['data']['cutoff'],
            include_hydrogen=self.config['data']['include_hydrogen']
        )
        
        test_dataset = PDBBindESA3DDataset(
            data_dir=self.config['data_dir'],
            split='test',
            max_atoms=self.config['data']['max_atoms'],
            cutoff=self.config['data']['cutoff'],
            include_hydrogen=self.config['data']['include_hydrogen']
        )
        
        # 创建collator
        collator = PDBBindESA3DCollator()
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            collate_fn=collator,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(
                node_features=batch.x,
                node_coords=batch.pos,
                edge_index=batch.edge_index,
                block_ids=batch.block_ids,
                batch=batch.batch,
                edge_attr=batch.edge_attr
            )
            
            # 计算损失
            loss = self.criterion(pred.squeeze(), batch.y.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self) -> Dict:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                # 前向传播
                pred = self.model(
                    node_features=batch.x,
                    node_coords=batch.pos,
                    edge_index=batch.edge_index,
                    block_ids=batch.block_ids,
                    batch=batch.batch,
                    edge_attr=batch.edge_attr
                )
                
                # 计算损失
                loss = self.criterion(pred.squeeze(), batch.y.squeeze())
                total_loss += loss.item()
                
                # 收集预测和真实值
                predictions.extend(pred.squeeze().cpu().numpy())
                targets.extend(batch.y.squeeze().cpu().numpy())
        
        # 计算指标
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        val_loss = total_loss / len(self.valid_loader)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 计算皮尔逊相关系数
        pearson_r, pearson_p = stats.pearsonr(targets, predictions)
        
        metrics = {
            'val_loss': val_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }
        
        return metrics
    
    def test(self) -> Dict:
        """测试模型"""
        self.model.eval()
        predictions = []
        targets = []
        pdb_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                # 前向传播
                pred = self.model(
                    node_features=batch.x,
                    node_coords=batch.pos,
                    edge_index=batch.edge_index,
                    block_ids=batch.block_ids,
                    batch=batch.batch,
                    edge_attr=batch.edge_attr
                )
                
                # 收集预测和真实值
                predictions.extend(pred.squeeze().cpu().numpy())
                targets.extend(batch.y.squeeze().cpu().numpy())
                pdb_ids.extend(batch.pdb_id)
        
        # 计算指标
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 计算皮尔逊相关系数
        pearson_r, pearson_p = stats.pearsonr(targets, predictions)
        
        metrics = {
            'test_mse': mse,
            'test_mae': mae,
            'test_r2': r2,
            'test_pearson_r': pearson_r,
            'test_pearson_p': pearson_p
        }
        
        # 保存预测结果
        results = {
            'pdb_ids': pdb_ids,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'metrics': metrics
        }
        
        with open(os.path.join(self.config['save_dir'], 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return metrics
    
    def train(self):
        """完整的训练过程"""
        print(f"Starting training with {len(self.train_loader)} training batches...")
        print(f"Validation set has {len(self.valid_loader)} batches")
        print(f"Test set has {len(self.test_loader)} batches")
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            self.val_losses.append(val_loss)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, "
                  f"R2: {val_metrics['r2']:.4f}, Pearson: {val_metrics['pearson_r']:.4f}")
            
            # 记录训练指标
            self.logger.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_metrics['mae'],
                'val_r2': val_metrics['r2'],
                'val_pearson_r': val_metrics['pearson_r'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step=epoch + 1)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }, os.path.join(self.config['save_dir'], 'best_model.pth'))
                
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['training']['patience']:
                    print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        # 加载最佳模型进行测试
        print("\nLoading best model for testing...")
        checkpoint = torch.load(os.path.join(self.config['save_dir'], 'best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试
        test_metrics = self.test()
        print(f"\nTest Results:")
        print(f"MSE: {test_metrics['test_mse']:.4f}")
        print(f"MAE: {test_metrics['test_mae']:.4f}")
        print(f"R2: {test_metrics['test_r2']:.4f}")
        print(f"Pearson: {test_metrics['test_pearson_r']:.4f}")
        
        # 记录测试结果到本地日志
        self.logger.log(test_metrics, step=None)
        
        # 绘制训练曲线和保存报告
        self._plot_training_curves()
        self.logger.save_metrics_summary()
        
        return test_metrics
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        # 使用LocalLogger绘制训练曲线
        plot_path = os.path.join(self.config['save_dir'], 'training_curves.png')
        self.logger.plot_training_curves(plot_path)


def main():
    parser = argparse.ArgumentParser(description='Train ESA-3D model on PDBBind dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 覆盖命令行参数
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.save_dir:
        config['save_dir'] = args.save_dir
    if args.device:
        config['device'] = args.device
    if args.debug:
        config['training']['num_epochs'] = 2
    
    # 创建并运行训练器
    trainer = ESA3DTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
