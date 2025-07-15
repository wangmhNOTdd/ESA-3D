# ESA-3D 快速开始指南

## 1. 环境设置

### 安装依赖
```bash
pip install torch torch-geometric torch-scatter torch-sparse
pip install numpy scipy scikit-learn matplotlib seaborn pandas tqdm
```

### 验证安装
```bash
cd ESA-3D
python test_model.py
```

## 2. 数据准备

### 准备PDBBind数据
```bash
# 下载PDBBind数据集到指定目录
# 例如：c:/path/to/pdbbind/

# 预处理数据
python preprocess.py \
    --pdbbind_dir "c:/path/to/pdbbind" \
    --output_dir "data/processed" \
    --max_atoms 1000 \
    --train_ratio 0.8 \
    --valid_ratio 0.1 \
    --test_ratio 0.1
```

### 数据结构
```
data/processed/
├── train.json      # 训练数据
├── valid.json      # 验证数据
├── test.json       # 测试数据
└── splits.json     # 数据分割信息
```

## 3. 模型训练

### 基础训练
```bash
python train.py --config config/default.json
```

### 自定义训练
```bash
python train.py \
    --config config/default.json \
    --data_dir "data/processed" \
    --save_dir "experiments/my_experiment" \
    --device "cuda"
```

### 调试模式
```bash
python train.py --config config/default.json --debug
```

## 4. 配置文件

### 模型配置 (config/default.json)
```json
{
  "model": {
    "node_dim": 41,        // 节点特征维度
    "edge_dim": 16,        // 边特征维度
    "hidden_dim": 128,     // 隐藏层维度
    "num_layers": 6,       // ESA-3D层数
    "num_heads": 8,        // 注意力头数
    "num_radial": 64,      // 径向基函数数量
    "cutoff": 10.0,        // 边的截断距离
    "num_seeds": 32,       // 池化种子数量
    "dropout": 0.1
  },
  "training": {
    "batch_size": 16,
    "num_epochs": 200,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "patience": 30
  }
}
```

## 5. 快速测试

### 测试单个分子
```python
import torch
from models.esa3d import ESA3DModel

# 创建模型
model = ESA3DModel(
    node_dim=41, edge_dim=16, hidden_dim=128,
    num_layers=6, num_heads=8, output_dim=1
)

# 准备数据
node_features = torch.randn(20, 41)  # 20个原子
node_coords = torch.randn(20, 3)     # 3D坐标
edge_index = torch.randint(0, 20, (2, 50))  # 50条边
block_ids = torch.randint(0, 5, (20,))      # 5个区块
edge_attr = torch.randn(50, 16)             # 边特征

# 预测
output = model(node_features, node_coords, edge_index, block_ids, edge_attr=edge_attr)
print(f"预测结果: {output}")
```

### 批量预测
```python
from torch_geometric.data import Data, Batch

# 创建批量数据
data_list = []
for i in range(4):  # 4个分子
    data = Data(
        x=torch.randn(15, 41),
        pos=torch.randn(15, 3),
        edge_index=torch.randint(0, 15, (2, 30)),
        edge_attr=torch.randn(30, 16),
        block_ids=torch.randint(0, 3, (15,))
    )
    data_list.append(data)

batch = Batch.from_data_list(data_list)
output = model(batch.x, batch.pos, batch.edge_index, batch.block_ids, batch.batch, batch.edge_attr)
print(f"批量预测结果: {output}")
```

## 6. 性能监控

### 本地可视化
```python
from utils.utils import plot_predictions, plot_training_curves

# 绘制预测结果
plot_predictions(predictions, targets, "ESA-3D Results")

# 绘制训练曲线
plot_training_curves(train_losses, val_losses)
```

### 实时监控
```python
# 训练过程中保存指标
import json

metrics_history = {
    'train_losses': [],
    'val_losses': [],
    'val_metrics': []
}

# 在训练循环中
metrics_history['train_losses'].append(train_loss)
metrics_history['val_losses'].append(val_loss)
metrics_history['val_metrics'].append(val_metrics)

# 保存到文件
with open('experiments/default/metrics_history.json', 'w') as f:
    json.dump(metrics_history, f, indent=2)
```

## 7. 模型评估

### 加载模型
```python
checkpoint = torch.load('experiments/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 计算指标
```python
from utils.utils import calculate_metrics

metrics = calculate_metrics(predictions, targets)
print(f"MSE: {metrics['mse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"Pearson: {metrics['pearson_r']:.4f}")
```

## 8. 常见问题

### Q: 模型输出NaN
A: 检查输入数据的范围，确保坐标不会过大；调整学习率；使用梯度裁剪。

### Q: 内存不足
A: 减少batch_size；减少max_atoms；使用梯度累积。

### Q: 训练速度慢
A: 使用更小的模型；减少num_layers；并行化数据加载。

### Q: 等变性测试失败
A: 检查坐标变换是否正确；确保模型使用相对位置向量。

## 9. 扩展功能

### 自定义数据集
```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # 加载你的数据
        pass
    
    def __getitem__(self, idx):
        # 返回PyG Data对象
        pass
```

### 多任务学习
```python
# 修改输出维度
model = ESA3DModel(output_dim=3)  # 3个任务

# 使用多任务损失
loss = F.mse_loss(pred[:, 0], target1) + F.mse_loss(pred[:, 1], target2)
```

### 预训练模型
```python
# 保存预训练模型
torch.save(model.state_dict(), 'pretrained_esa3d.pth')

# 加载预训练模型
model.load_state_dict(torch.load('pretrained_esa3d.pth'))
```

## 10. 参考资源

- [ESA论文](https://arxiv.org/abs/xxxx.xxxxx)
- [GET论文](https://arxiv.org/abs/xxxx.xxxxx)
- [PyTorch Geometric文档](https://pytorch-geometric.readthedocs.io/)
- [PDBBind数据集](http://www.pdbbind.org.cn/)

## 支持

如遇问题，请参考：
1. 运行 `python test_model.py` 检查环境
2. 查看 `experiments/` 目录下的日志文件
3. 检查数据格式是否正确
4. 调整模型超参数
