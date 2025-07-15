# ESA-3D: Edge-Set Attention for 3D Molecular Property Prediction

ESA-3D是一个基于边集合注意力的3D分子性质预测模型，结合了ESA（Edge-Set Attention）的简洁性和GET（Generalist Equivariant Transformer）的等变性。

## 数据准备

### 方法1：使用PDBBind数据集（推荐）

1. **下载PDBBind数据** ：
   ```bash
   # 显示下载说明
   python setup_data.py download
   ```

2. **预处理数据** ：
   ```bash
   # 自动预处理PDBBind数据
   python setup_data.py preprocess
   ```

### 方法2：使用示例数据（快速测试）

```bash
# 创建示例数据用于测试
python setup_data.py sample

# 使用示例数据训练
python train.py --config config/default.json --data_dir data/sample
```

### 注意事项

- 处理后的数据文件较大，已在`.gitignore`中排除
- 请在本地生成数据，不要提交到Git仓库
- 示例数据仅用于测试代码功能

## 核心思想

### 1. 边中心表示（Edge-Centric Representation）
- 将分子图视为**边的集合**而非节点的集合
- 每条边包含两个原子的信息，比单个节点信息更丰富
- 边的等变特征：相对位置向量 `r_ij = x_i - x_j`

### 2. 等变边注意力（Equivariant Edge Attention）
- **不变特征更新**：通过多头注意力机制更新边的化学特征
- **等变特征更新**：通过门控机制更新边的几何特征，保持E(3)等变性
- **几何感知**：注意力权重同时考虑化学特征、距离和方向信息

### 3. 交错注意力模式（Interleaved Attention Patterns）
- **区块内注意力（Intra-Block）**：同一残基内的边之间进行注意力计算
- **区块间注意力（Inter-Block）**：不同残基之间的边进行注意力计算
- **层次化学习**：先局部细化，再全局交互

## 架构设计

### 模型组件

1. **EdgeEmbedding**: 将节点特征和边特征合并为边的初始表示
2. **ESA3DEncoder**: 多层交错的等变边注意力
3. **AttentionPooling**: 注意力池化，将边特征聚合为图表示
4. **输出层**: MLP回归头

### 核心模块

```python
class ESA3DBlock(nn.Module):
    """ESA-3D基本块"""
    def forward(self, edge_features, edge_coords, intra_mask, inter_mask):
        # 1. 区块内注意力（向内看）
        intra_features, intra_coords = self.intra_attention(
            edge_features, edge_coords, intra_mask
        )
        
        # 2. 区块间注意力（向外看）
        inter_features, inter_coords = self.inter_attention(
            intra_features, intra_coords, inter_mask
        )
        
        # 3. 前馈网络
        return self.ffn(inter_features), inter_coords
```

## 数据格式

### 输入数据
```python
{
    'node_features': torch.Tensor,  # [N, node_dim] 节点特征
    'node_coords': torch.Tensor,    # [N, 3] 节点坐标
    'edge_index': torch.Tensor,     # [2, E] 边索引
    'block_ids': torch.Tensor,      # [N] 节点的区块ID
    'edge_attr': torch.Tensor,      # [E, edge_dim] 边特征(可选)
    'batch': torch.Tensor,          # [N] 批次信息
}
```

### 特征构建
- **节点特征**: 原子类型 + 残基类型 + 原子位置编码
- **边特征**: 基于距离的径向基函数特征
- **区块ID**: 残基标识符，用于构建intra/inter掩码

## 使用方法

### 1. 数据预处理
```python
from data.pdbbind_dataset import preprocess_pdbbind_data

# 创建数据分割
splits = create_data_splits(pdbbind_dir)

# 预处理数据
preprocess_pdbbind_data(
    pdbbind_dir="path/to/pdbbind",
    output_dir="data/processed",
    splits=splits
)
```

### 2. 模型训练
```python
from train import ESA3DTrainer

# 加载配置
with open('config/default.json', 'r') as f:
    config = json.load(f)

# 创建训练器
trainer = ESA3DTrainer(config)

# 开始训练
trainer.train()
```

### 3. 模型推理
```python
from models.esa3d import ESA3DModel

# 加载模型
model = ESA3DModel(...)
checkpoint = torch.load('path/to/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 推理
output = model(node_features, node_coords, edge_index, block_ids)
```

## 实验设置

### 默认配置
```json
{
  "model": {
    "node_dim": 41,      // 节点特征维度
    "edge_dim": 16,      // 边特征维度
    "hidden_dim": 128,   // 隐藏层维度
    "num_layers": 6,     // 层数
    "num_heads": 8,      // 注意力头数
    "num_radial": 64,    // 径向基函数数量
    "cutoff": 10.0,      // 边的截断距离
    "num_seeds": 32,     // 池化种子数量
    "dropout": 0.1       // Dropout率
  },
  "training": {
    "batch_size": 16,
    "num_epochs": 200,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001
  }
}
```

### 性能指标
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **Pearson相关系数**: 预测与真实值的相关性

## 理论基础

### 等变性保证
ESA-3D通过以下方式保证E(3)等变性：
1. **相对位置向量**: 边的等变特征是相对位置 `r_ij = x_i - x_j`
2. **等变更新**: 坐标更新遵循 `x'_i = x_i + Σ α_ij * gate(v_j) * (x_i - x_j)`
3. **不变聚合**: 最终池化使用坐标的范数作为不变特征

### 与GET的关系
ESA-3D可以看作GET的边中心版本：
- **GET**: 双层注意力（原子内 + 原子间）
- **ESA-3D**: 交错注意力（区块内 + 区块间）
- **共同点**: 都实现了无损的多尺度几何建模

### 与ESA的关系
ESA-3D扩展了原始ESA到3D几何场景：
- **ESA**: 基于线图的masked/vanilla注意力
- **ESA-3D**: 基于3D几何的intra/inter注意力
- **共同点**: 都采用边中心表示和交错注意力模式

## 文件结构

```
ESA-3D/
├── models/
│   └── esa3d.py                    # 主模型
├── modules/
│   └── equivariant_edge_attention.py  # 等变边注意力
├── data/
│   └── pdbbind_dataset.py          # 数据处理
├── config/
│   └── default.json                # 配置文件
├── train.py                        # 训练脚本
├── test_model.py                   # 模型测试
└── README.md                       # 说明文档
```

## 运行示例

### 训练模型
```bash
cd ESA-3D
python train.py --config config/default.json
```

### 测试模型
```bash
python test_model.py
```

### 数据预处理
```bash
python data/pdbbind_dataset.py
```

## 关键特性

1. **端到端训练**: 无需复杂的预处理或位置编码
2. **E(3)等变性**: 对3D旋转和平移保持不变
3. **多尺度建模**: 同时捕获局部和全局信息
4. **高效实现**: 基于PyTorch和PyTorch Geometric
5. **可扩展性**: 易于扩展到其他3D分子任务

## 未来改进

1. **多模态融合**: 结合序列、结构和动力学信息
2. **预训练策略**: 大规模无监督预训练
3. **动态建模**: 处理分子构象变化
4. **计算优化**: 稀疏注意力和梯度累积
5. **领域适应**: 迁移学习到新的分子数据集

## 引用

如果使用ESA-3D，请引用：
```bibtex
@article{esa3d2024,
  title={ESA-3D: Edge-Set Attention for 3D Molecular Property Prediction},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

MIT License
