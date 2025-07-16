#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
多粒度建图数据集
支持原子级、残基级、混合级建图，类似GET项目的思想
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, add_self_loops

from .pdbbind_dataset import COMMON_ELEMENTS, COMMON_RESIDUES, COMMON_ATOM_NAMES, load_complex_data


class Block:
    """表示一个残基或分子片段"""
    
    def __init__(self, symbol: str, atoms: List[Dict], coords: np.ndarray):
        """
        Args:
            symbol: 残基名称或分子片段类型
            atoms: 包含的原子列表
            coords: 原子坐标 [N, 3]
        """
        self.symbol = symbol
        self.atoms = atoms
        self.coords = coords
        self.num_atoms = len(atoms)
        
    def get_center(self) -> np.ndarray:
        """获取block的中心坐标"""
        return np.mean(self.coords, axis=0)
    
    def get_radius(self) -> float:
        """获取block的半径"""
        center = self.get_center()
        distances = np.linalg.norm(self.coords - center, axis=1)
        return np.max(distances)
    
    def __len__(self):
        return self.num_atoms
    
    def __repr__(self):
        return f"Block({self.symbol}, {self.num_atoms} atoms)"


class MultiGranularityDataset(Dataset):
    """支持多粒度建图的数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        granularity: str = 'atom',  # 'atom', 'residue', 'mixed'
        max_atoms: int = 1000,
        max_residues: int = 200,
        atom_cutoff: float = 5.0,
        residue_cutoff: float = 10.0,
        add_self_loops: bool = True,
        include_hydrogen: bool = False,
    ):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据集分割 ('train', 'valid', 'test')
            granularity: 建图粒度 ('atom', 'residue', 'mixed')
            max_atoms: 最大原子数量
            max_residues: 最大残基数量
            atom_cutoff: 原子级边的截断距离
            residue_cutoff: 残基级边的截断距离
            add_self_loops: 是否添加自环
            include_hydrogen: 是否包含氢原子
        """
        self.data_dir = data_dir
        self.split = split
        self.granularity = granularity
        self.max_atoms = max_atoms
        self.max_residues = max_residues
        self.atom_cutoff = atom_cutoff
        self.residue_cutoff = residue_cutoff
        self.add_self_loops = add_self_loops
        self.include_hydrogen = include_hydrogen
        
        # 加载数据
        self.data_list = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载预处理的数据"""
        data_file = os.path.join(self.data_dir, f'{self.split}.json')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            data_list = json.load(f)
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        # 根据粒度构建不同的图
        if self.granularity == 'atom':
            graph_data = self._build_atom_graph(data_item)
        elif self.granularity == 'residue':
            graph_data = self._build_residue_graph(data_item)
        elif self.granularity == 'mixed':
            graph_data = self._build_mixed_graph(data_item)
        else:
            raise ValueError(f"Unknown granularity: {self.granularity}")
        
        return graph_data
    
    def _build_atom_graph(self, data_item: Dict) -> Data:
        """构建原子级图"""
        atoms = data_item['atoms']
        coords = np.array(data_item['coords'])
        
        # 过滤氢原子
        if not self.include_hydrogen:
            non_h_indices = [i for i, atom in enumerate(atoms) if atom['element'] != 'H']
            atoms = [atoms[i] for i in non_h_indices]
            coords = coords[non_h_indices]
        
        # 限制原子数量
        if len(atoms) > self.max_atoms:
            atoms = atoms[:self.max_atoms]
            coords = coords[:self.max_atoms]
        
        # 构建节点特征
        node_features = self._build_atom_features(atoms)
        
        # 构建边和边特征
        edge_index, edge_attr = self._build_atom_edges(coords)
        
        # 构建block_ids (原子级建图时，每个原子是一个block)
        block_ids = torch.arange(len(atoms), dtype=torch.long)
        
        # 创建PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(coords, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            block_ids=block_ids,
            y=torch.tensor([data_item['label']], dtype=torch.float),
            pdb_id=data_item['pdb_id'],
        )
        
        return data
    
    def _build_residue_graph(self, data_item: Dict) -> Data:
        """构建残基级图"""
        atoms = data_item['atoms']
        coords = np.array(data_item['coords'])
        
        # 过滤氢原子
        if not self.include_hydrogen:
            non_h_indices = [i for i, atom in enumerate(atoms) if atom['element'] != 'H']
            atoms = [atoms[i] for i in non_h_indices]
            coords = coords[non_h_indices]
        
        # 将原子按残基分组
        blocks = self._group_atoms_by_residue(atoms, coords)
        
        # 限制残基数量
        if len(blocks) > self.max_residues:
            blocks = blocks[:self.max_residues]
        
        # 构建残基级节点特征
        node_features = self._build_residue_features(blocks)
        
        # 构建残基级边
        edge_index, edge_attr = self._build_residue_edges(blocks)
        
        # 构建残基中心坐标
        residue_coords = np.array([block.get_center() for block in blocks])
        
        # block_ids (残基级建图时，每个残基是一个block)
        block_ids = torch.arange(len(blocks), dtype=torch.long)
        
        # 创建PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(residue_coords, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            block_ids=block_ids,
            y=torch.tensor([data_item['label']], dtype=torch.float),
            pdb_id=data_item['pdb_id'],
        )
        
        return data
    
    def _build_mixed_graph(self, data_item: Dict) -> Data:
        """构建混合级图（原子+残基）"""
        atoms = data_item['atoms']
        coords = np.array(data_item['coords'])
        
        # 过滤氢原子
        if not self.include_hydrogen:
            non_h_indices = [i for i, atom in enumerate(atoms) if atom['element'] != 'H']
            atoms = [atoms[i] for i in non_h_indices]
            coords = coords[non_h_indices]
        
        # 限制原子数量
        if len(atoms) > self.max_atoms:
            atoms = atoms[:self.max_atoms]
            coords = coords[:self.max_atoms]
        
        # 将原子按残基分组
        blocks = self._group_atoms_by_residue(atoms, coords)
        
        # 限制残基数量
        if len(blocks) > self.max_residues:
            blocks = blocks[:self.max_residues]
        
        # 构建混合图：原子作为节点，残基作为block
        all_atoms = []
        all_coords = []
        block_ids = []
        
        for block_id, block in enumerate(blocks):
            all_atoms.extend(block.atoms)
            all_coords.extend(block.coords)
            block_ids.extend([block_id] * len(block.atoms))
        
        # 构建原子级节点特征
        node_features = self._build_atom_features(all_atoms)
        
        # 构建原子级边
        coords_array = np.array(all_coords)
        edge_index, edge_attr = self._build_atom_edges(coords_array)
        
        # 创建PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(coords_array, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            block_ids=torch.tensor(block_ids, dtype=torch.long),
            y=torch.tensor([data_item['label']], dtype=torch.float),
            pdb_id=data_item['pdb_id'],
        )
        
        return data
    
    def _group_atoms_by_residue(self, atoms: List[Dict], coords: np.ndarray) -> List[Block]:
        """将原子按残基分组"""
        residue_groups = {}
        
        for i, atom in enumerate(atoms):
            residue_id = atom.get('residue_id', 'UNK')
            residue_name = atom.get('residue', 'UNK')
            
            if residue_id not in residue_groups:
                residue_groups[residue_id] = {
                    'atoms': [],
                    'coords': [],
                    'residue_name': residue_name
                }
            
            residue_groups[residue_id]['atoms'].append(atom)
            residue_groups[residue_id]['coords'].append(coords[i])
        
        # 创建Block对象
        blocks = []
        for residue_id, group in residue_groups.items():
            block = Block(
                symbol=group['residue_name'],
                atoms=group['atoms'],
                coords=np.array(group['coords'])
            )
            blocks.append(block)
        
        return blocks
    
    def _build_atom_features(self, atoms: List[Dict]) -> np.ndarray:
        """构建原子级特征"""
        features = []
        
        for atom in atoms:
            # 元素特征
            element_features = self._get_element_features(atom['element'])
            
            # 残基特征
            residue_features = self._get_residue_features(atom.get('residue', 'UNK'))
            
            # 原子位置特征
            atom_pos_features = self._get_atom_position_features(atom.get('atom_name', 'UNK'))
            
            # 合并所有特征
            atom_features = np.concatenate([
                element_features,
                residue_features,
                atom_pos_features,
            ])
            
            features.append(atom_features)
        
        return np.array(features)
    
    def _build_residue_features(self, blocks: List[Block]) -> np.ndarray:
        """构建残基级特征"""
        features = []
        
        for block in blocks:
            # 残基类型特征
            residue_features = self._get_residue_features(block.symbol)
            
            # 残基大小特征
            size_features = np.array([
                len(block.atoms),  # 原子数量
                block.get_radius(),  # 半径
            ])
            
            # 原子组成特征
            element_counts = np.zeros(len(COMMON_ELEMENTS))
            for atom in block.atoms:
                element = atom['element']
                if element in COMMON_ELEMENTS:
                    element_counts[COMMON_ELEMENTS.index(element)] += 1
            
            # 归一化
            if element_counts.sum() > 0:
                element_counts = element_counts / element_counts.sum()
            
            # 合并所有特征
            block_features = np.concatenate([
                residue_features,
                size_features,
                element_counts,
            ])
            
            features.append(block_features)
        
        return np.array(features)
    
    def _build_atom_edges(self, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建原子级边"""
        num_atoms = len(coords)
        
        # 计算距离矩阵
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        
        # 找到cutoff范围内的边
        edge_indices = np.where((distances < self.atom_cutoff) & (distances > 0))
        
        edge_index = torch.tensor(np.stack(edge_indices), dtype=torch.long)
        
        # 边属性：距离特征
        edge_distances = distances[edge_indices]
        edge_attr = self._get_edge_features(edge_distances)
        
        # 确保无向图
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # 添加自环
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_atoms
            )
        
        return edge_index, edge_attr
    
    def _build_residue_edges(self, blocks: List[Block]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建残基级边"""
        num_blocks = len(blocks)
        
        # 计算残基中心之间的距离
        centers = np.array([block.get_center() for block in blocks])
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        
        # 找到cutoff范围内的边
        edge_indices = np.where((distances < self.residue_cutoff) & (distances > 0))
        
        edge_index = torch.tensor(np.stack(edge_indices), dtype=torch.long)
        
        # 边属性：距离特征
        edge_distances = distances[edge_indices]
        edge_attr = self._get_edge_features(edge_distances)
        
        # 确保无向图
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # 添加自环
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_blocks
            )
        
        return edge_index, edge_attr
    
    def _get_element_features(self, element: str) -> np.ndarray:
        """获取元素特征"""
        features = np.zeros(len(COMMON_ELEMENTS) + 1)  # +1 for unknown
        
        if element in COMMON_ELEMENTS:
            features[COMMON_ELEMENTS.index(element)] = 1.0
        else:
            features[-1] = 1.0  # unknown
        
        return features
    
    def _get_residue_features(self, residue: str) -> np.ndarray:
        """获取残基特征"""
        features = np.zeros(len(COMMON_RESIDUES) + 1)  # +1 for ligand/unknown
        
        if residue in COMMON_RESIDUES:
            features[COMMON_RESIDUES.index(residue)] = 1.0
        else:
            features[-1] = 1.0  # ligand or unknown
        
        return features
    
    def _get_atom_position_features(self, atom_name: str) -> np.ndarray:
        """获取原子位置特征"""
        features = np.zeros(len(COMMON_ATOM_NAMES) + 1)  # +1 for unknown
        
        if atom_name in COMMON_ATOM_NAMES:
            features[COMMON_ATOM_NAMES.index(atom_name)] = 1.0
        else:
            features[-1] = 1.0  # unknown
        
        return features
    
    def _get_edge_features(self, distances: np.ndarray) -> torch.Tensor:
        """获取边特征"""
        # 距离的高斯径向基函数编码
        num_rbf = 16
        cutoff = max(self.atom_cutoff, self.residue_cutoff)
        
        # 高斯中心
        centers = np.linspace(0, cutoff, num_rbf)
        width = cutoff / num_rbf
        
        # 计算RBF特征
        rbf_features = np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * width ** 2))
        
        return torch.tensor(rbf_features, dtype=torch.float)


def create_multi_granularity_config(granularity: str = 'mixed') -> Dict:
    """创建多粒度配置"""
    
    base_config = {
        "exp_name": f"esa3d_pdbbind_{granularity}",
        "data_dir": "./data/processed",
        "save_dir": f"./experiments/{granularity}",
        "device": "cuda",
        "random_seed": 42,
        
        "data": {
            "granularity": granularity,
            "max_atoms": 1000,
            "max_residues": 200,
            "atom_cutoff": 5.0,
            "residue_cutoff": 10.0,
            "include_hydrogen": False
        },
        
        "training": {
            "batch_size": 4,
            "num_epochs": 200,
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "patience": 30,
            "num_workers": 4
        }
    }
    
    # 根据粒度调整模型配置
    if granularity == 'atom':
        base_config["model"] = {
            "node_dim": 43,  # element(11) + residue(21) + atom_pos(11)
            "edge_dim": 16,
            "hidden_dim": 128,
            "num_layers": 6,
            "num_heads": 8,
            "num_radial": 64,
            "cutoff": 5.0,
            "num_seeds": 32,
            "output_dim": 1,
            "dropout": 0.1
        }
    elif granularity == 'residue':
        base_config["model"] = {
            "node_dim": 33,  # residue(21) + size(2) + element_counts(10)
            "edge_dim": 16,
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
            "num_radial": 64,
            "cutoff": 10.0,
            "num_seeds": 16,
            "output_dim": 1,
            "dropout": 0.1
        }
    elif granularity == 'mixed':
        base_config["model"] = {
            "node_dim": 43,  # element(11) + residue(21) + atom_pos(11)
            "edge_dim": 16,
            "hidden_dim": 128,
            "num_layers": 6,
            "num_heads": 8,
            "num_radial": 64,
            "cutoff": 5.0,
            "num_seeds": 32,
            "output_dim": 1,
            "dropout": 0.1
        }
    
    return base_config


if __name__ == "__main__":
    # 测试多粒度数据集
    data_dir = "./data/processed"
    
    for granularity in ['atom', 'residue', 'mixed']:
        print(f"\n测试 {granularity} 粒度建图...")
        
        try:
            dataset = MultiGranularityDataset(
                data_dir=data_dir,
                split='train',
                granularity=granularity,
                max_atoms=100,
                max_residues=50,
            )
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  节点数: {sample.x.shape[0]}")
                print(f"  边数: {sample.edge_index.shape[1]}")
                print(f"  节点特征维度: {sample.x.shape[1]}")
                print(f"  边特征维度: {sample.edge_attr.shape[1]}")
                print(f"  Block IDs: {sample.block_ids.shape}")
            else:
                print("  数据集为空")
        except Exception as e:
            print(f"  错误: {e}")
