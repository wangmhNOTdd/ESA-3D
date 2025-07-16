好的，我们来详细拆解ESA和我们构想的ESA-3D的原理，并给出每一步的公式。

---

### **Part 1: ESA 原始模型的详细原理与公式**

ESA的核心是将图学习的中心从**节点**转移到**边**，并使用交错的**Masked自注意力**和**标准自注意力**来学习边的表示。

#### **1. 输入表示 (Input Representation)**

给定一个图 `G=(V, E)`，其中节点有特征 $n_i \in \mathbb{R}^{d_n}$，边有特征 $e_{ij} \in \mathbb{R}^{d_e}$。

* **核心操作：为每条边创建特征向量**
  对于连接节点`i`和`j`的边`e_ij`，其初始特征向量 $x_{ij}$ 是由源节点、目标节点和边自身特征拼接而成：
  $ x_{ij} = [n_i \ || \ n_j \ || \ e_{ij}] \in \mathbb{R}^{2d_n + d_e} $
  所有边的特征向量构成一个集合（或矩阵）$X \in \mathbb{R}^{N_e \times (2d_n + d_e)}$，其中 $N_e$ 是边的总数。

#### **2. 编码器 (Encoder)**

编码器由两种模块交错组成：**Masked Self-Attention Block (MAB)** 和 **Self-Attention Block (SAB)**。

##### **2.1 Masked Self-Attention Block (MAB)**

MAB是ESA的核心，它限制了注意力只在有连接关系的边之间发生。

1. **输入**: 边的特征矩阵 $X \in \mathbb{R}^{N_e \times d_{in}}$。

2. **层归一化 (Pre-LayerNorm)**:
   $ X_{norm} = \text{LayerNorm}(X) $

3. **生成Query, Key, Value**:
   $ Q = X_{norm} W_Q, \quad K = X_{norm} W_K, \quad V = X_{norm} W_V $
   其中 $W_Q, W_K, W_V \in \mathbb{R}^{d_{in} \times d_{model}}$ 是可学习的权重矩阵。

4. **计算边-边连接Mask ($M_{edge}$)**:
   这是关键一步。Mask矩阵 $M_{edge} \in \mathbb{R}^{N_e \times N_e}$ 定义了哪些边对可以相互关注。如果边`p`和边`q`在原始图中共享一个节点，则 $M_{edge}[p, q] = 0$，否则 $M_{edge}[p, q] = -\infty$。
   （具体实现见论文`Algorithm 1`，通过检查源/目标节点是否相同来高效计算）。

5. **Masked Scaled Dot-Product Attention**:
   $ \text{AttentionScores} = \frac{QK^T}{\sqrt{d_k}} + M_{edge} $
   $ \text{AttentionWeights} = \text{softmax}(\text{AttentionScores}) $
   $ H_{attn} = \text{AttentionWeights} \cdot V $
   这里 $d_k$ 是Key向量的维度。该过程通常在多头（Multi-Head）下并行进行。

6. **残差连接与前馈网络**:
   $ H_{res} = X + H_{attn} $
   $ X_{out} = H_{res} + \text{MLP}(\text{LayerNorm}(H_{res})) $

##### **2.2 Self-Attention Block (SAB)**

SAB与MAB的**唯一区别**在于它不使用Mask矩阵，或者说使用的Mask矩阵全为0。这允许任意两条边之间进行全局信息交互。

* **公式**:
  $ \text{SAB}(X) = \text{MAB}(X, \mathbf{0}) $
  其中 $\mathbf{0}$ 是一个全零矩阵。

#### **3. 池化模块 (Pooling by Multi-Head Attention, PMA)**

PMA是一个可学习的、将边表示聚合成图表示的模块。

1. **输入**: 编码器输出的边特征矩阵 $Z \in \mathbb{R}^{N_e \times d_{model}}$。
2. **引入可学习的种子向量**: 初始化一个种子矩阵 $S_k \in \mathbb{R}^{k \times d_{model}}$，其中 `k` 是种子数量（一个超参数，如32）。
3. **交叉注意力 (Cross-Attention)**:

   * 种子作为Query，边的特征作为Key和Value。
     $ Q_{seed} = S_k W_Q', \quad K_{edge} = Z W_K', \quad V_{edge} = Z W_V' $
     $ H_{pool} = \text{softmax}\left(\frac{Q_{seed} K_{edge}^T}{\sqrt{d_k}}\right) V_{edge} \in \mathbb{R}^{k \times d_{model}} $
4. **后续处理**: 聚合到种子上的信息 $H_{pool}$ 可以再经过几层SAB进行内部信息交互，最后将`k`个种子向量求和或平均，得到最终的图表示 $H_{graph}$。
   $ H_{graph} = \text{mean}(\text{SAB}*p(H*{pool})) $

---

### **Part 2: ESA-3D 构想模型的详细原理与公式**

ESA-3D将ESA的哲学扩展到3D几何空间，核心是引入E(3)等变性。

#### **1. 输入表示 (Input Representation)**

给定一个复合物，我们关注其**边的集合**。对于连接原子`i`和`j`的边`p`：

* **不变特征 $h_p \in \mathbb{R}^{d_h}$**: 由原子`i`和`j`的化学类型等标量特征拼接或处理得到。
* **等变特征 $\vec{x}_p \in \mathbb{R}^{3}$**: 定义为相对位置矢量 $\vec{r}_{ij} = \vec{x}_i - \vec{x}_j$。

#### **2. 编码器 (Encoder)**

编码器由我们设计的两种**等变边注意力 (Equivariant Edge Attention, EEA)** 模块交错组成：**EEA-Intra** 和 **EEA-Inter**。

##### **一个通用的等变边注意力层 (EEA Layer) 的公式**

假设输入是边的集合 ${h_p, \vec{x}_p}$ 和一个Mask矩阵 `M`。

1. **生成不变的Query, Key, Value**:
   $ q_p = W_Q h_p, \quad k_p = W_K h_p, \quad v_p = W_V h_p $

2. **计算不变的注意力权重 $\alpha_{pq}$**:

   * **几何信息编码**:
     $ \vec{d}_{pq} = \vec{x}*p - \vec{x}*q $ (两条边的相对矢量)
     $ dist*{pq} = ||\vec{d}*{pq}||_2 $ (相对距离)
   * **注意力分数**:
     $ score_{pq} = \text{MLP}*{attn}(h_p, h_q, \text{RBF}(dist*{pq})) $ (这里也可以加入点积等更多不变几何量)
   * **Masking与归一化**:
     $ \alpha_{pq} = \text{softmax}*q(score*{pq} + M_{pq}) $

3. **等变信息更新 (Equivariant Update)**:

   * **不变特征更新 (Scalar Update)**:
     $ \Delta h_p = \sum_q \alpha_{pq} \cdot \text{MLP}_{h}(v_p, v_q) $
     $ h'_p = h_p + \Delta h_p $
   * **等变特征更新 (Vector Update)**: (借鉴MACE和GET)
     $ \Delta \vec{x}*p = \sum_q \alpha*{pq} \cdot (\sigma_x(v_p, v_q) \cdot \vec{d}_{pq}) $
     其中 $\sigma_x$ 是一个MLP，将不变特征映射为一个标量门控信号。
     $ \vec{x}'_p = \vec{x}_p + \Delta \vec{x}_p $

##### **2.1 EEA-Intra (区块内注意力)**

* **实现**: 使用**区块内Mask ($M_{intra}$)** 的EEA层。
  $({h'}, {\vec{x'}}) = \text{EEA}({h}, {\vec{x}}, M_{intra})$
  $M_{intra}[p, q] = 0$ 如果边`p`和`q`在同一区块且共享节点，否则为`-\infty`。

##### **2.2 EEA-Inter (区块间注意力)**

* **实现**: 使用**区块间Mask ($M_{inter}$)** 的EEA层。
  $ ({h''}, {\vec{x''}}) = \text{EEA}({h'}, {\vec{x'}}, M_{inter}) $
  $M_{inter}[p, q] = 0$ 如果边`p`和`q`在不同区块，否则为`-\infty`。

#### **3. 池化与预测 (Pooling & Prediction)**

1. **边到节点的等变聚合 (Edge-to-Node Equivariant Aggregation)**:

   * 在L层编码器后，我们有最终的边表示 `{h_p, \vec{x}_p}`。
   * 对于每个原子`i`，聚合所有与之相连的边的信息：
     $ h_i^{node} = \sum_{j \in N(i)} \text{MLP}*{node_h}(h*{ij}) $
     $ \vec{x}*i^{node} = \sum*{j \in N(i)} \sigma_{node_x}(h_{ij}) \cdot \vec{x}_{ij} $

2. **全局不变池化 (Global Invariant Pooling)**:

   * 计算所有节点不变特征的均值：
     $ H_{graph} = \frac{1}{N} \sum_i h_i^{node} $
   * 计算所有节点等变特征范数的均值：
     $ X_{graph} = \frac{1}{N} \sum_i ||\vec{x}_i^{node}||_2 $

3. **最终预测**:

   * 将全局不变特征拼接起来，送入MLP：
     $ \text{Affinity} = \text{MLP}([H_{graph} \ || \ X_{graph}]) $


