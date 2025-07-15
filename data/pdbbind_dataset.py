#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, add_self_loops

# 简化的原子词汇表，不依赖GET项目
COMMON_ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
COMMON_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
COMMON_ATOM_NAMES = ['CA', 'CB', 'CG', 'CD', 'CE', 'CF', 'N', 'O', 'S', 'P']


class PDBBindESA3DDataset(Dataset):
    """PDBBind数据集的ESA-3D适配版本"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_atoms: int = 1000,
        cutoff: float = 10.0,
        add_self_loops: bool = True,
        include_hydrogen: bool = False,
    ):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据集分割 ('train', 'valid', 'test')
            max_atoms: 最大原子数量
            cutoff: 边的截断距离
            add_self_loops: 是否添加自环
            include_hydrogen: 是否包含氢原子
        """
        self.data_dir = data_dir
        self.split = split
        self.max_atoms = max_atoms
        self.cutoff = cutoff
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
        
        # 构建PyG Data对象
        graph_data = self._build_graph(data_item)
        
        return graph_data
    
    def _build_graph(self, data_item: Dict) -> Data:
        """从数据项构建图数据"""
        
        # 获取原子信息
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
        
        num_atoms = len(atoms)
        
        # 构建节点特征
        node_features = self._build_node_features(atoms)
        
        # 构建边和边特征
        edge_index, edge_attr = self._build_edges(coords)
        
        # 构建区块ID (残基ID)
        block_ids = self._build_block_ids(atoms)
        
        # 获取标签
        label = float(data_item['label'])
        
        # 构建PyG Data对象
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            pos=torch.tensor(coords, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            block_ids=torch.tensor(block_ids, dtype=torch.long),
            y=torch.tensor([label], dtype=torch.float),
            num_atoms=num_atoms,
            pdb_id=data_item.get('pdb_id', 'unknown'),
        )
        
        return data
    
    def _build_node_features(self, atoms: List[Dict]) -> np.ndarray:
        """构建节点特征"""
        features = []
        
        for atom in atoms:
            # 原子类型的one-hot编码
            element = atom['element']
            atom_type_feat = self._get_atom_type_features(element)
            
            # 残基信息
            residue_feat = self._get_residue_features(atom.get('residue', 'UNK'))
            
            # 原子位置编码
            atom_pos_feat = self._get_atom_position_features(atom.get('atom_name', 'UNK'))
            
            # 合并特征
            atom_features = np.concatenate([atom_type_feat, residue_feat, atom_pos_feat])
            features.append(atom_features)
        
        return np.array(features)
    
    def _get_atom_type_features(self, element: str) -> np.ndarray:
        """获取原子类型特征"""
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
    
    def _build_edges(self, coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于距离构建边"""
        num_atoms = len(coords)
        
        # 计算距离矩阵
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        
        # 找到cutoff范围内的边
        edge_indices = np.where((distances < self.cutoff) & (distances > 0))
        
        edge_index = torch.tensor(np.stack(edge_indices), dtype=torch.long)
        
        # 边属性：距离特征
        edge_distances = distances[edge_indices]
        edge_attr = self._get_edge_features(edge_distances)
        
        # 确保无向图 (to_undirected会自动处理edge_attr的复制)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # 添加自环
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_atoms
            )
        
        return edge_index, edge_attr
    
    def _get_edge_features(self, distances: np.ndarray) -> torch.Tensor:
        """获取边特征"""
        # 使用高斯径向基函数编码距离
        num_rbf = 16
        centers = np.linspace(0, self.cutoff, num_rbf)
        widths = np.ones(num_rbf) * 0.5
        
        rbf_features = np.exp(-widths * (distances[:, None] - centers[None, :]) ** 2)
        
        return torch.tensor(rbf_features, dtype=torch.float)
    
    def _build_block_ids(self, atoms: List[Dict]) -> List[int]:
        """构建区块ID (残基ID)"""
        block_ids = []
        
        current_block_id = 0
        last_residue = None
        
        for atom in atoms:
            residue_id = atom.get('residue_id', 'UNK')
            
            if residue_id != last_residue:
                if last_residue is not None:
                    current_block_id += 1
                last_residue = residue_id
            
            block_ids.append(current_block_id)
        
        return block_ids


class PDBBindESA3DCollator:
    """ESA-3D数据的批处理器"""
    
    def __init__(self, follow_batch: Optional[List[str]] = None):
        self.follow_batch = follow_batch or []
    
    def __call__(self, batch: List[Data]) -> Batch:
        """批处理数据"""
        return Batch.from_data_list(batch, follow_batch=self.follow_batch)


def preprocess_pdbbind_data(
    pdbbind_dir: str,
    output_dir: str,
    splits: Dict[str, List[str]],
    max_atoms: int = 1000,
    include_hydrogen: bool = False,
):
    """
    预处理PDBBind数据
    
    Args:
        pdbbind_dir: PDBBind数据目录
        output_dir: 输出目录
        splits: 数据分割信息 {'train': [...], 'valid': [...], 'test': [...]}
        max_atoms: 最大原子数量
        include_hydrogen: 是否包含氢原子
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载亲和力数据
    affinity_file = os.path.join(pdbbind_dir, 'metadata', 'affinities.json')
    affinity_data = {}
    
    if os.path.exists(affinity_file):
        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)
        print(f"加载了 {len(affinity_data)} 个亲和力数据")
    else:
        print(f"警告: 亲和力数据文件不存在: {affinity_file}")
    
    # 处理每个分割
    for split_name, pdb_ids in splits.items():
        print(f"Processing {split_name} split...")
        
        processed_data = []
        
        for pdb_id in pdb_ids:
            try:
                # 加载蛋白质-配体复合物
                complex_data = load_complex_data(pdbbind_dir, pdb_id, include_hydrogen)
                
                if complex_data is None:
                    continue
                
                # 获取亲和力标签
                label = affinity_data.get(pdb_id, 0.0)
                
                # 限制原子数量
                if len(complex_data['atoms']) > max_atoms:
                    complex_data['atoms'] = complex_data['atoms'][:max_atoms]
                    complex_data['coords'] = complex_data['coords'][:max_atoms]
                
                # 添加到处理后的数据
                processed_item = {
                    'pdb_id': pdb_id,
                    'atoms': complex_data['atoms'],
                    'coords': complex_data['coords'].tolist(),
                    'label': label,
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                continue
        
        # 保存处理后的数据
        output_file = os.path.join(output_dir, f'{split_name}.json')
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Saved {len(processed_data)} items to {output_file}")


def load_complex_data(pdbbind_dir: str, pdb_id: str, include_hydrogen: bool = False) -> Optional[Dict]:
    """加载蛋白质-配体复合物数据"""
    
    # 查找复合物文件
    complex_file = None
    
    # 实际的PDB文件在 pdb_files/ 子目录中
    pdb_files_dir = os.path.join(pdbbind_dir, 'pdb_files')
    
    possible_paths = [
        os.path.join(pdb_files_dir, pdb_id, f'{pdb_id}.pdb'),
        os.path.join(pdb_files_dir, pdb_id, f'{pdb_id}_pocket.pdb'),
        os.path.join(pdb_files_dir, pdb_id, f'{pdb_id}_fixed.pdb'),
        os.path.join(pdb_files_dir, pdb_id, f'{pdb_id}_complex.pdb'),
        os.path.join(pdb_files_dir, pdb_id, f'{pdb_id}_protein.pdb'),
        # 也尝试原始路径以防万一
        os.path.join(pdbbind_dir, pdb_id, f'{pdb_id}_complex.pdb'),
        os.path.join(pdbbind_dir, pdb_id, f'{pdb_id}_protein.pdb'),
        os.path.join(pdbbind_dir, 'refined-set', pdb_id, f'{pdb_id}_complex.pdb'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            complex_file = path
            break
    
    if complex_file is None:
        # 调试信息：看看实际有什么文件
        pdb_dir = os.path.join(pdb_files_dir, pdb_id)
        if os.path.exists(pdb_dir):
            files = os.listdir(pdb_dir)
            print(f"Available files in {pdb_dir}: {files}")
        return None
    
    # 解析PDB文件
    try:
        atoms = []
        coords = []
        
        with open(complex_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # 解析原子信息
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    residue_id = line[21:26].strip()
                    element = line[76:78].strip()
                    
                    if not element:
                        element = atom_name[0]
                    
                    # 过滤氢原子
                    if not include_hydrogen and element == 'H':
                        continue
                    
                    # 解析坐标
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atoms.append({
                        'atom_name': atom_name,
                        'residue': residue_name,
                        'residue_id': residue_id,
                        'element': element,
                    })
                    
                    coords.append([x, y, z])
        
        if len(atoms) == 0:
            return None
        
        return {
            'atoms': atoms,
            'coords': np.array(coords),
        }
        
    except Exception as e:
        print(f"Error parsing {complex_file}: {e}")
        return None


def create_data_splits(
    pdbbind_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """创建数据分割"""
    
    # 获取所有PDB ID
    all_pdb_ids = []
    
    # PDB文件实际存储在 pdb_files/ 子目录中
    pdb_files_dir = os.path.join(pdbbind_dir, 'pdb_files')
    
    if not os.path.exists(pdb_files_dir):
        print(f"错误: PDB文件目录不存在: {pdb_files_dir}")
        return {'train': [], 'valid': [], 'test': []}
    
    # 查找所有可用的PDB ID
    for item in os.listdir(pdb_files_dir):
        item_path = os.path.join(pdb_files_dir, item)
        if os.path.isdir(item_path) and len(item) == 4:
            all_pdb_ids.append(item)
    
    print(f"找到 {len(all_pdb_ids)} 个PDB分子")
    
    if len(all_pdb_ids) == 0:
        print("警告: 没有找到任何PDB分子")
        return {'train': [], 'valid': [], 'test': []}
    
    # 随机分割
    np.random.seed(random_seed)
    np.random.shuffle(all_pdb_ids)
    
    num_total = len(all_pdb_ids)
    num_train = int(num_total * train_ratio)
    num_valid = int(num_total * valid_ratio)
    
    splits = {
        'train': all_pdb_ids[:num_train],
        'valid': all_pdb_ids[num_train:num_train + num_valid],
        'test': all_pdb_ids[num_train + num_valid:],
    }
    
    return splits


if __name__ == "__main__":
    # 示例使用
    pdbbind_dir = "c:/Users/18778/Desktop/torch-learn/GET/datasets/PDBBind/pdbbind"
    output_dir = "c:/Users/18778/Desktop/torch-learn/ESA-3D/data/processed"
    
    # 创建数据分割
    splits = create_data_splits(pdbbind_dir)
    
    # 预处理数据
    preprocess_pdbbind_data(
        pdbbind_dir=pdbbind_dir,
        output_dir=output_dir,
        splits=splits,
        max_atoms=1000,
        include_hydrogen=False,
    )
    
    print("Data preprocessing completed!")
