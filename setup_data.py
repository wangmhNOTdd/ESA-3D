#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备脚本
用于下载和预处理ESA-3D训练数据
"""
import os
import sys
import subprocess
from pathlib import Path

def download_pdbbind_data():
    """下载PDBBind数据集"""
    print("PDBBind数据集下载说明:")
    print("1. 请访问 http://www.pdbbind.org.cn/ 注册并下载PDBBind数据集")
    print("2. 将下载的数据解压到 GET/datasets/PDBBind/pdbbind/ 目录")
    print("3. 确保目录结构如下:")
    print("   GET/datasets/PDBBind/pdbbind/")
    print("   ├── metadata/")
    print("   │   ├── affinities.json")
    print("   │   └── ...")
    print("   └── pdb_files/")
    print("       ├── 10gs/")
    print("       ├── 1a28/")
    print("       └── ...")
    print("")

def preprocess_data():
    """预处理数据"""
    print("开始预处理数据...")
    
    # 检查PDBBind数据是否存在
    pdbbind_path = Path("../GET/datasets/PDBBind/pdbbind")
    if not pdbbind_path.exists():
        print(f"错误: PDBBind数据目录不存在: {pdbbind_path}")
        print("请先下载PDBBind数据集。")
        return False
    
    # 运行预处理脚本
    cmd = [
        sys.executable, "preprocess.py",
        "--pdbbind_dir", str(pdbbind_path),
        "--output_dir", "data/processed",
        "--max_atoms", "200",  # 使用更小的原子数
        "--train_ratio", "0.8",
        "--valid_ratio", "0.1",
        "--test_ratio", "0.1"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"预处理失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def create_sample_data():
    """创建示例数据用于测试"""
    print("创建示例数据...")
    
    # 创建一个小的示例数据集
    sample_data = {
        "pdb_id": "1abc",
        "atoms": [
            {"atom_name": "CA", "residue": "ALA", "residue_id": "1", "element": "C"},
            {"atom_name": "CB", "residue": "ALA", "residue_id": "1", "element": "C"},
            {"atom_name": "N", "residue": "ALA", "residue_id": "1", "element": "N"},
        ],
        "coords": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "label": 5.0
    }
    
    os.makedirs("data/sample", exist_ok=True)
    
    import json
    for split in ["train", "valid", "test"]:
        with open(f"data/sample/{split}.json", "w") as f:
            json.dump([sample_data] * 10, f, indent=2)
    
    print("示例数据创建完成: data/sample/")

def main():
    print("ESA-3D 数据准备脚本")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            download_pdbbind_data()
        elif sys.argv[1] == "preprocess":
            preprocess_data()
        elif sys.argv[1] == "sample":
            create_sample_data()
        else:
            print("未知命令。使用: python setup_data.py [download|preprocess|sample]")
    else:
        print("使用方法:")
        print("  python setup_data.py download    # 显示下载说明")
        print("  python setup_data.py preprocess  # 预处理数据")
        print("  python setup_data.py sample      # 创建示例数据")

if __name__ == "__main__":
    main()
