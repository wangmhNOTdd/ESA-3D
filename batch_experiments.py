#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
批量实验脚本
比较不同粒度和注意力机制的性能
"""
import os
import json
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def run_experiment(config_path: str, granularity: str, attention_type: str, model_size: str):
    """运行单个实验"""
    
    cmd = [
        'python', 'train_multigranularity.py',
        '--config', config_path,
        '--granularity', granularity,
        '--attention_type', attention_type,
        '--model_size', model_size,
    ]
    
    print(f"运行实验: {granularity} + {attention_type} + {model_size}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            print(f"实验成功完成")
            return True, result.stdout
        else:
            print(f"实验失败: {result.stderr}")
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"实验超时")
        return False, "实验超时"
    except Exception as e:
        print(f"实验异常: {e}")
        return False, str(e)


def load_experiment_results(exp_dir: str) -> Dict:
    """加载实验结果"""
    results_file = os.path.join(exp_dir, 'test_results.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    
    return None


def collect_all_results(experiments: List[Dict]) -> pd.DataFrame:
    """收集所有实验结果"""
    results = []
    
    for exp in experiments:
        exp_name = f"esa3d_{exp['granularity']}_{exp['attention_type']}_{exp['model_size']}"
        exp_dir = f"./experiments/{exp_name}"
        
        result = load_experiment_results(exp_dir)
        
        if result:
            results.append({
                'granularity': exp['granularity'],
                'attention_type': exp['attention_type'],
                'model_size': exp['model_size'],
                'test_loss': result['test_loss'],
                'test_mae': result['test_mae'],
                'test_rmse': result['test_rmse'],
                'exp_name': exp_name,
                'status': 'success'
            })
        else:
            results.append({
                'granularity': exp['granularity'],
                'attention_type': exp['attention_type'],
                'model_size': exp['model_size'],
                'test_loss': None,
                'test_mae': None,
                'test_rmse': None,
                'exp_name': exp_name,
                'status': 'failed'
            })
    
    return pd.DataFrame(results)


def visualize_results(results_df: pd.DataFrame, save_dir: str):
    """可视化实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 只保留成功的实验
    success_df = results_df[results_df['status'] == 'success'].copy()
    
    if len(success_df) == 0:
        print("没有成功的实验结果可视化")
        return
    
    # 1. 不同粒度的性能对比
    plt.figure(figsize=(12, 8))
    
    metrics = ['test_loss', 'test_mae', 'test_rmse']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=success_df, x='granularity', y=metric)
        plt.title(f'{metric.upper()} by Granularity')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'granularity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 不同注意力机制的性能对比
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=success_df, x='attention_type', y=metric)
        plt.title(f'{metric.upper()} by Attention Type')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'attention_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 不同模型大小的性能对比
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=success_df, x='model_size', y=metric)
        plt.title(f'{metric.upper()} by Model Size')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 综合热力图
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        
        # 创建透视表
        pivot_table = success_df.pivot_table(
            values=metric,
            index='granularity',
            columns='attention_type',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r')
        plt.title(f'{metric.upper()} Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 性能排名
    plt.figure(figsize=(12, 8))
    
    success_df_sorted = success_df.sort_values('test_mae')
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(success_df_sorted)), success_df_sorted['test_mae'])
    plt.xlabel('Experiment')
    plt.ylabel('Test MAE')
    plt.title('Test MAE Ranking')
    plt.xticks(range(len(success_df_sorted)), success_df_sorted['exp_name'], rotation=45)
    
    # 用颜色区分粒度
    colors = {'atom': 'red', 'residue': 'blue', 'mixed': 'green'}
    for i, (idx, row) in enumerate(success_df_sorted.iterrows()):\n        bars[i].set_color(colors[row['granularity']])\n    \n    plt.legend(handles=[plt.Rectangle((0,0),1,1, color=colors[g]) for g in colors.keys()], \n               labels=colors.keys(), title='Granularity')\n    \n    plt.tight_layout()\n    plt.savefig(os.path.join(save_dir, 'performance_ranking.png'), dpi=300, bbox_inches='tight')\n    plt.close()\n    \n    print(f\"可视化结果保存在: {save_dir}\")\n\n\ndef generate_summary_report(results_df: pd.DataFrame, save_dir: str):\n    \"\"\"生成总结报告\"\"\"\n    os.makedirs(save_dir, exist_ok=True)\n    \n    success_df = results_df[results_df['status'] == 'success'].copy()\n    failed_df = results_df[results_df['status'] == 'failed'].copy()\n    \n    report = []\n    report.append(\"# ESA-3D 多粒度实验结果报告\")\n    report.append(\"\")\n    report.append(f\"## 实验概况\")\n    report.append(f\"- 总实验数: {len(results_df)}\")\n    report.append(f\"- 成功实验数: {len(success_df)}\")\n    report.append(f\"- 失败实验数: {len(failed_df)}\")\n    report.append(f\"- 成功率: {len(success_df)/len(results_df)*100:.1f}%\")\n    report.append(\"\")\n    \n    if len(success_df) > 0:\n        report.append(\"## 最佳结果\")\n        \n        # 找到最佳结果\n        best_mae = success_df.loc[success_df['test_mae'].idxmin()]\n        best_rmse = success_df.loc[success_df['test_rmse'].idxmin()]\n        best_loss = success_df.loc[success_df['test_loss'].idxmin()]\n        \n        report.append(f\"### 最佳 MAE: {best_mae['test_mae']:.4f}\")\n        report.append(f\"- 实验: {best_mae['exp_name']}\")\n        report.append(f\"- 粒度: {best_mae['granularity']}\")\n        report.append(f\"- 注意力: {best_mae['attention_type']}\")\n        report.append(f\"- 模型大小: {best_mae['model_size']}\")\n        report.append(\"\")\n        \n        report.append(f\"### 最佳 RMSE: {best_rmse['test_rmse']:.4f}\")\n        report.append(f\"- 实验: {best_rmse['exp_name']}\")\n        report.append(f\"- 粒度: {best_rmse['granularity']}\")\n        report.append(f\"- 注意力: {best_rmse['attention_type']}\")\n        report.append(f\"- 模型大小: {best_rmse['model_size']}\")\n        report.append(\"\")\n        \n        # 按粒度分组的统计\n        report.append(\"## 按粒度分组的平均性能\")\n        report.append(\"\")\n        granularity_stats = success_df.groupby('granularity')[['test_loss', 'test_mae', 'test_rmse']].agg(['mean', 'std'])\n        \n        for granularity in granularity_stats.index:\n            report.append(f\"### {granularity.capitalize()} 级建图\")\n            report.append(f\"- MAE: {granularity_stats.loc[granularity, ('test_mae', 'mean')]:.4f} ± {granularity_stats.loc[granularity, ('test_mae', 'std')]:.4f}\")\n            report.append(f\"- RMSE: {granularity_stats.loc[granularity, ('test_rmse', 'mean')]:.4f} ± {granularity_stats.loc[granularity, ('test_rmse', 'std')]:.4f}\")\n            report.append(f\"- Loss: {granularity_stats.loc[granularity, ('test_loss', 'mean')]:.4f} ± {granularity_stats.loc[granularity, ('test_loss', 'std')]:.4f}\")\n            report.append(\"\")\n        \n        # 按注意力机制分组的统计\n        report.append(\"## 按注意力机制分组的平均性能\")\n        report.append(\"\")\n        attention_stats = success_df.groupby('attention_type')[['test_loss', 'test_mae', 'test_rmse']].agg(['mean', 'std'])\n        \n        for attention in attention_stats.index:\n            report.append(f\"### {attention.capitalize()} 注意力\")\n            report.append(f\"- MAE: {attention_stats.loc[attention, ('test_mae', 'mean')]:.4f} ± {attention_stats.loc[attention, ('test_mae', 'std')]:.4f}\")\n            report.append(f\"- RMSE: {attention_stats.loc[attention, ('test_rmse', 'mean')]:.4f} ± {attention_stats.loc[attention, ('test_rmse', 'std')]:.4f}\")\n            report.append(f\"- Loss: {attention_stats.loc[attention, ('test_loss', 'mean')]:.4f} ± {attention_stats.loc[attention, ('test_loss', 'std')]:.4f}\")\n            report.append(\"\")\n    \n    if len(failed_df) > 0:\n        report.append(\"## 失败实验\")\n        report.append(\"\")\n        for idx, row in failed_df.iterrows():\n            report.append(f\"- {row['exp_name']} ({row['granularity']} + {row['attention_type']} + {row['model_size']})\")\n        report.append(\"\")\n    \n    # 保存报告\n    with open(os.path.join(save_dir, 'summary_report.md'), 'w', encoding='utf-8') as f:\n        f.write('\\n'.join(report))\n    \n    print(f\"总结报告保存在: {os.path.join(save_dir, 'summary_report.md')}\")\n\n\ndef main():\n    # 实验配置\n    experiments = [\n        # 不同粒度\n        {'granularity': 'atom', 'attention_type': 'sparse', 'model_size': 'small'},\n        {'granularity': 'atom', 'attention_type': 'sparse', 'model_size': 'medium'},\n        {'granularity': 'residue', 'attention_type': 'sparse', 'model_size': 'small'},\n        {'granularity': 'residue', 'attention_type': 'sparse', 'model_size': 'medium'},\n        {'granularity': 'mixed', 'attention_type': 'sparse', 'model_size': 'small'},\n        {'granularity': 'mixed', 'attention_type': 'sparse', 'model_size': 'medium'},\n        \n        # 不同注意力机制\n        {'granularity': 'mixed', 'attention_type': 'block', 'model_size': 'medium'},\n        {'granularity': 'mixed', 'attention_type': 'full', 'model_size': 'small'},\n        \n        # 大模型测试\n        {'granularity': 'mixed', 'attention_type': 'sparse', 'model_size': 'large'},\n    ]\n    \n    config_path = './config/multigranularity.json'\n    \n    # 检查配置文件\n    if not os.path.exists(config_path):\n        print(f\"配置文件不存在: {config_path}\")\n        return\n    \n    # 运行实验\n    print(f\"开始运行 {len(experiments)} 个实验...\")\n    \n    for i, exp in enumerate(experiments):\n        print(f\"\\n=== 实验 {i+1}/{len(experiments)} ===\")\n        \n        success, output = run_experiment(\n            config_path,\n            exp['granularity'],\n            exp['attention_type'],\n            exp['model_size']\n        )\n        \n        if success:\n            print(\"✅ 实验成功\")\n        else:\n            print(\"❌ 实验失败\")\n            print(f\"错误信息: {output}\")\n        \n        # 等待一会儿\n        time.sleep(5)\n    \n    # 收集结果\n    print(\"\\n收集实验结果...\")\n    results_df = collect_all_results(experiments)\n    \n    # 保存结果\n    results_dir = './experiments/batch_results'\n    os.makedirs(results_dir, exist_ok=True)\n    \n    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)\n    \n    # 可视化结果\n    print(\"生成可视化结果...\")\n    visualize_results(results_df, results_dir)\n    \n    # 生成总结报告\n    print(\"生成总结报告...\")\n    generate_summary_report(results_df, results_dir)\n    \n    print(f\"\\n批量实验完成! 结果保存在: {results_dir}\")\n    \n    # 打印简要结果\n    print(\"\\n=== 简要结果 ===\")\n    success_df = results_df[results_df['status'] == 'success']\n    if len(success_df) > 0:\n        best_result = success_df.loc[success_df['test_mae'].idxmin()]\n        print(f\"最佳结果: {best_result['exp_name']}\")\n        print(f\"MAE: {best_result['test_mae']:.4f}\")\n        print(f\"RMSE: {best_result['test_rmse']:.4f}\")\n    else:\n        print(\"没有成功的实验\")\n\n\nif __name__ == \"__main__\":\n    main()
