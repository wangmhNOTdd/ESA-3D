#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
本地性能监控工具，替代wandb
"""
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional


class LocalLogger:
    """本地训练日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str = "default"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.jsonl")
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 存储指标历史
        self.metrics_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'timestamps': [],
            'epochs': []
        }
        
        # 记录实验开始时间
        self.start_time = time.time()
        
        print(f"LocalLogger initialized: {self.log_file}")
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置到文件"""
        config_path = os.path.join(self.log_dir, f"{self.experiment_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_path}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步数或epoch
        """
        # 添加时间戳
        log_entry = {
            'timestamp': time.time(),
            'step': step,
            'elapsed_time': time.time() - self.start_time,
            **metrics
        }
        
        # 写入文件
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # 更新历史记录
        if step is not None:
            self.metrics_history['epochs'].append(step)
            self.metrics_history['timestamps'].append(log_entry['timestamp'])
            
            if 'train_loss' in metrics:
                self.metrics_history['train_losses'].append(metrics['train_loss'])
            if 'val_loss' in metrics:
                self.metrics_history['val_losses'].append(metrics['val_loss'])
            if 'val_mae' in metrics and 'val_r2' in metrics:
                self.metrics_history['val_metrics'].append({
                    'mae': metrics['val_mae'],
                    'r2': metrics['val_r2'],
                    'pearson_r': metrics.get('val_pearson_r', 0)
                })
        
        # 实时显示重要指标
        self._print_metrics(metrics, step)
    
    def _print_metrics(self, metrics: Dict[str, Any], step: Optional[int]):
        """打印指标到控制台"""
        if step is not None:
            print(f"Step {step}: ", end="")
        
        important_metrics = ['train_loss', 'val_loss', 'val_mae', 'val_r2', 'val_pearson_r']
        metric_strs = []
        
        for key in important_metrics:
            if key in metrics:
                if isinstance(metrics[key], float):
                    metric_strs.append(f"{key}: {metrics[key]:.4f}")
                else:
                    metric_strs.append(f"{key}: {metrics[key]}")
        
        if metric_strs:
            print(" | ".join(metric_strs))
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """绘制训练曲线"""
        if not self.metrics_history['epochs']:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = self.metrics_history['epochs']
        
        # 损失曲线
        if self.metrics_history['train_losses'] and self.metrics_history['val_losses']:
            axes[0, 0].plot(epochs, self.metrics_history['train_losses'], 
                           label='Train Loss', color='blue')
            axes[0, 0].plot(epochs, self.metrics_history['val_losses'], 
                           label='Val Loss', color='red')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 损失曲线（对数坐标）
        if self.metrics_history['train_losses'] and self.metrics_history['val_losses']:
            axes[0, 1].plot(epochs, self.metrics_history['train_losses'], 
                           label='Train Loss', color='blue')
            axes[0, 1].plot(epochs, self.metrics_history['val_losses'], 
                           label='Val Loss', color='red')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss (log scale)')
            axes[0, 1].set_title('Training and Validation Loss (Log Scale)')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # MAE曲线
        if self.metrics_history['val_metrics']:
            mae_values = [m['mae'] for m in self.metrics_history['val_metrics']]
            axes[1, 0].plot(epochs, mae_values, label='MAE', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].set_title('Validation MAE')
            axes[1, 0].grid(True, alpha=0.3)
        
        # R²曲线
        if self.metrics_history['val_metrics']:
            r2_values = [m['r2'] for m in self.metrics_history['val_metrics']]
            axes[1, 1].plot(epochs, r2_values, label='R²', color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].set_title('Validation R²')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {save_path}")
        
        plt.show()
    
    def save_metrics_summary(self, save_path: Optional[str] = None):
        """保存指标摘要"""
        if save_path is None:
            save_path = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': len(self.metrics_history['epochs']),
            'total_time': time.time() - self.start_time,
            'metrics_history': self.metrics_history
        }
        
        # 计算最佳指标
        if self.metrics_history['val_losses']:
            best_epoch = np.argmin(self.metrics_history['val_losses'])
            summary['best_epoch'] = int(self.metrics_history['epochs'][best_epoch])
            summary['best_val_loss'] = float(self.metrics_history['val_losses'][best_epoch])
            
            if self.metrics_history['val_metrics']:
                best_metrics = self.metrics_history['val_metrics'][best_epoch]
                summary['best_val_mae'] = float(best_metrics['mae'])
                summary['best_val_r2'] = float(best_metrics['r2'])
                summary['best_val_pearson_r'] = float(best_metrics.get('pearson_r', 0))
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Metrics summary saved to: {save_path}")
        return summary
    
    def load_from_file(self, log_file: str):
        """从文件加载日志"""
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    step = entry.get('step')
                    if step is not None:
                        self.metrics_history['epochs'].append(step)
                        self.metrics_history['timestamps'].append(entry['timestamp'])
                        
                        if 'train_loss' in entry:
                            self.metrics_history['train_losses'].append(entry['train_loss'])
                        if 'val_loss' in entry:
                            self.metrics_history['val_losses'].append(entry['val_loss'])
                        if 'val_mae' in entry and 'val_r2' in entry:
                            self.metrics_history['val_metrics'].append({
                                'mae': entry['val_mae'],
                                'r2': entry['val_r2'],
                                'pearson_r': entry.get('val_pearson_r', 0)
                            })
        
        print(f"Loaded {len(self.metrics_history['epochs'])} entries from {log_file}")


def create_experiment_report(log_dir: str, experiment_name: str, output_file: str = None):
    """创建实验报告"""
    logger = LocalLogger(log_dir, experiment_name)
    log_file = os.path.join(log_dir, f"{experiment_name}.jsonl")
    
    if os.path.exists(log_file):
        logger.load_from_file(log_file)
        
        # 生成图表
        plot_path = os.path.join(log_dir, f"{experiment_name}_curves.png")
        logger.plot_training_curves(plot_path)
        
        # 保存摘要
        summary_path = os.path.join(log_dir, f"{experiment_name}_summary.json")
        summary = logger.save_metrics_summary(summary_path)
        
        # 打印摘要
        print("\n" + "="*50)
        print(f"Experiment Report: {experiment_name}")
        print("="*50)
        print(f"Total Epochs: {summary.get('total_epochs', 0)}")
        print(f"Total Time: {summary.get('total_time', 0):.2f} seconds")
        print(f"Best Epoch: {summary.get('best_epoch', 'N/A')}")
        print(f"Best Val Loss: {summary.get('best_val_loss', 'N/A'):.4f}")
        print(f"Best Val MAE: {summary.get('best_val_mae', 'N/A'):.4f}")
        print(f"Best Val R²: {summary.get('best_val_r2', 'N/A'):.4f}")
        print(f"Best Val Pearson: {summary.get('best_val_pearson_r', 'N/A'):.4f}")
        print("="*50)
        
        return summary
    else:
        print(f"Log file not found: {log_file}")
        return None


if __name__ == "__main__":
    # 示例使用
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate experiment report')
    parser.add_argument('--log_dir', type=str, default='experiments/default/logs', 
                       help='Log directory')
    parser.add_argument('--experiment_name', type=str, default='default',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    create_experiment_report(args.log_dir, args.experiment_name)
