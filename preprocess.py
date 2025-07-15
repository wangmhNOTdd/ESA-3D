#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
简单的数据预处理脚本，用于快速开始ESA-3D训练
"""
import os
import json
import argparse
from data.pdbbind_dataset import create_data_splits, preprocess_pdbbind_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess PDBBind data for ESA-3D')
    parser.add_argument('--pdbbind_dir', type=str, required=True, help='PDBBind数据目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--max_atoms', type=int, default=1000, help='最大原子数量')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--include_hydrogen', action='store_true', help='是否包含氢原子')
    
    args = parser.parse_args()
    
    print(f"PDBBind目录: {args.pdbbind_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大原子数: {args.max_atoms}")
    print(f"数据分割: {args.train_ratio:.1f}/{args.valid_ratio:.1f}/{args.test_ratio:.1f}")
    print(f"包含氢原子: {args.include_hydrogen}")
    
    # 检查输入目录
    if not os.path.exists(args.pdbbind_dir):
        print(f"错误: PDBBind目录不存在: {args.pdbbind_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据分割
    print("\n创建数据分割...")
    splits = create_data_splits(
        args.pdbbind_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    print(f"训练集: {len(splits['train'])} 个分子")
    print(f"验证集: {len(splits['valid'])} 个分子")
    print(f"测试集: {len(splits['test'])} 个分子")
    
    # 保存分割信息
    with open(os.path.join(args.output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    # 预处理数据
    print("\n开始预处理数据...")
    preprocess_pdbbind_data(
        pdbbind_dir=args.pdbbind_dir,
        output_dir=args.output_dir,
        splits=splits,
        max_atoms=args.max_atoms,
        include_hydrogen=args.include_hydrogen
    )
    
    print("\n数据预处理完成!")
    print(f"处理后的数据保存在: {args.output_dir}")
    print("可以开始训练模型了。")


if __name__ == "__main__":
    main()
