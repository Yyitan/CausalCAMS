# CausalCAMS 技术文档

> **基于结构因果模型的组合零样本学习（Step 1：因果解耦验证）**
> 在 CAMS 基线上引入 SCM + CIB 因果解耦，不含扩散模型

---

## 1. 动机与目标

CAMS 通过 Gated Cross-Attention 和 Multi-Space Disentanglement 将图像特征分解为属性空间 S_a、对象空间 S_o 和组合空间，并用交叉熵监督各分支。然而，这种解耦**仅依赖分类监督**，没有显式约束 S_a 和 S_o 的统计独立性。当模型学习到"红色（属性）→ 苹果（对象）"这样的共现偏见时，S_a 和 S_o 之间仍然存在统计相关性，导致模型对未见组合泛化能力不足。

**目标**：在 CAMS 基础上引入因果结构约束，使 S_a、S_o 在统计意义上独立，从而提升未见组合的泛化能力。

本文档对应 **Step 1**：只加入 SCM 框架 + CIB 因果解耦，不包含扩散模型。

---

## 2. 结构因果模型（SCM）设计

### 2.1 变量定义

| 符号     | 含义                                   | 维度        |
|----------|----------------------------------------|-------------|
| X        | 输入图像                               | H × W × 3  |
| S_a      | **属性因果因子**（Y-causative for attr）  | d = 768     |
| S_o      | **对象因果因子**（Y-causative for obj）   | d = 768     |
| Z        | **非因果因子**（Y-non-causative）        | d = 768     |
| Y_attr   | 属性标签                               | 标量        |
| Y_obj    | 对象标签                               | 标量        |

### 2.2 因果关系图

```
S_a ──────────────────→ Y_attr
S_o ──────────────────→ Y_obj
S_a, S_o, Z ─────────→ X        （联合生成图像）

约束：
  S_a ⊥ S_o   （属性与对象独立，核心组合假设）
  S_a ⊥ Z     （属性与非因果噪声独立）
  S_o ⊥ Z     （对象与非因果噪声独立）
```

### 2.3 与原始 CAMS 的对应关系

```
CAMS 原始             CausalCAMS 映射
─────────────────     ────────────────────────
att (tr_a 输出)  →    S_a（属性因果因子）
obj (tr_o 输出)  →    S_o（对象因果因子）
com (tr_c 输出)  →    保留（组合分支，不参与因果约束）
glb（CLS token） →    保留（全局分支）
[新增]           →    Z（tr_z 输出，非因果因子）
```

Z 的作用是"吸收"与标签无关的背景、光照、纹理信息，使 S_a 和 S_o 更"纯粹"。Z 本身**不参与分类**，只参与因果约束。

---

## 3. Causal Information Bottleneck（CIB）

### 3.1 目标函数推导

最大化：

```
I(X, Y_attr, Y_obj ; S_a, S_o, Z)
= I(X; S_a, S_o, Z)
  + I(Y_attr; S_a)
  + I(Y_obj; S_o)
  - I(S_a; S_o)          ← 属性-对象解耦
  - I(S_a; Z)            ← 属性-非因果解耦
  - I(S_o; Z)            ← 对象-非因果解耦
  - λ · I(X; S_a, S_o, Z) ← 信息瓶颈
```

### 3.2 各项的具体实现

| 项目                   | 实现方式                                              |
|------------------------|-------------------------------------------------------|
| I(Y_attr; S_a)         | 属性分类交叉熵 L_CE_attr（CAMS 已有）                 |
| I(Y_obj; S_o)          | 对象分类交叉熵 L_CE_obj（CAMS 已有）                  |
| I(Y_comp; S_a, S_o)    | 组合分类交叉熵 L_CE_com（CAMS 已有）                  |
| I(S_a; S_o)            | **CLUB 上界估计**（新增）                             |
| I(S_a; Z)              | **CLUB 上界估计**（新增）                             |
| I(S_o; Z)              | **CLUB 上界估计**（新增）                             |
| λ · I(X; S_a, S_o, Z)  | KL 散度正则（可选，Step 1 通过 `use_kl` 控制）        |

### 3.3 CLUB 估计器（Contrastive Log-ratio Upper Bound）

对于任意两个随机变量 X, Y，CLUB 给出互信息上界：

```
I_CLUB(X; Y) = E_{p(x,y)}[log q_θ(y|x)] - E_{p(x)}E_{p(y)}[log q_θ(y|x)]
```

其中 q_θ(y|x) = N(μ_θ(x), σ²_θ(x)) 是一个可学习的条件高斯分布。

**在一个 batch 中的近似计算**：

```python
# 正样本项（对角线）
pos = mean_i [ log N(y_i; μ(x_i), σ²(x_i)) ]

# 负样本项（交叉对）
neg = mean_i [ mean_j [ log N(y_j; μ(x_i), σ²(x_i)) ] ]

I_CLUB(X; Y) ≈ pos - neg
```

**训练策略**：CLUB 估计器需要两步更新：

1. **训练估计器**（使上界更紧）：最大化对数似然
   - 对 S_a 的 detached 副本做 MLE
   - 仅更新估计器参数，不更新编码器

2. **训练编码器**（最小化互信息上界）：使 CLUB 值最小
   - 使用当前 S_a, S_o, Z 计算 CLUB 上界
   - 仅更新编码器参数

为避免两次 backward 的开销，本实现采用 **detach** 策略在单次 backward 中完成：

```python
# 估计器损失：用 detached 特征，只传梯度给估计器网络
club_learn = club_estimator.learning_loss(S_a.detach(), S_o.detach())

# 编码器损失：用真实特征，只传梯度给编码器
club_bound = club_estimator(S_a, S_o)

# 总损失（单次 backward）
loss = task_loss + η * club_bound + α * club_learn
```

---

## 4. 完整损失函数

```
L_total = α_att · L_CE_attr           (属性分支，CAMS 原有)
        + α_obj · L_CE_obj            (对象分支，CAMS 原有)
        + α_com · L_CE_com            (组合分支，CAMS 原有)
        + α_glb · L_CE_glb            (全局分支，CAMS 原有)
        + η_ao  · CLUB(S_a, S_o)      (属性-对象解耦，新增)
        + η_az  · CLUB(S_a, Z)        (属性-非因果解耦，新增)
        + η_oz  · CLUB(S_o, Z)        (对象-非因果解耦，新增)
        + α_club · L_club_learn        (CLUB 估计器训练，新增)
```

超参数建议初始值：
- η_ao = η_az = η_oz = 1e-3（从小值开始，避免解耦过度破坏分类）
- α_club = 1e-2（估计器学习率相对更大）

---

## 5. 模型架构修改

### 5.1 文件改动一览

```
CAMS/
├── clip_modules/
│   └── gca.py              ← 修改：新增 tr_z、proj_z，输出 Z
├── model/
│   ├── causal_modules.py   ← 新增：CLUBEstimator 类
│   ├── cams.py             ← 修改：实例化 CLUB 估计器，加入 CIB 损失
│   └── configure_model.py  ← 修改：将因果模块参数加入优化器
├── configs/
│   ├── mit-states.yml      ← 修改：新增因果超参数
│   ├── utzappos.yml        ← 修改：新增因果超参数
│   └── cgqa.yml            ← 修改：新增因果超参数
└── test.py                 ← 修改：解包 5 个输出
```

### 5.2 数据流图

```
图像 X
  │
  ▼
CLIP ViT 低层 Block（冻结）→ H_0
  │
  ▼ （最后 M 层，LoRA 微调）
Gated Cross-Attention + 高层 Block
  │
  ├── Latent Units Q̃ [B, K, 1024]
  │
  ├── tr_a(Q̃) → mean → @ proj_att → S_a [B, 768]  → 属性分类头
  ├── tr_o(Q̃) → mean → @ proj_obj → S_o [B, 768]  → 对象分类头
  ├── tr_c(Q̃) → mean → @ proj_com → C  [B, 768]   → 组合分类头
  └── tr_z(Q̃) → mean → @ proj_z  → Z  [B, 768]   → 不分类，仅参与 CLUB
  
  CLS token → @ proj → glb [B, 768] → 全局分类头

因果约束（CLUB）：
  CLUBNet_ao(S_a, S_o) → I_CLUB(S_a; S_o)
  CLUBNet_az(S_a, Z)   → I_CLUB(S_a; Z)
  CLUBNet_oz(S_o, Z)   → I_CLUB(S_o; Z)
```

---

## 6. 推理流程（不变）

推理时不使用 Z 和 CLUB，与原始 CAMS 完全一致：

```
s(a_i, o_j) = β · p_glb(i,j|X)
            + (1-β) · p_com(i,j|X)
            + p_att(i|X) · p_obj(j|X)
```

---

## 7. 消融实验设计

验证因果解耦效果的消融设置：

| 配置名               | 描述                           | 验证目标                    |
|----------------------|--------------------------------|-----------------------------|
| CAMS（baseline）     | 原始 CAMS，无因果约束          | 基线                        |
| +Z_branch            | 增加 Z 分支，无 CLUB           | Z 分支是否提供有效表示      |
| +CLUB_ao             | 只加 CLUB(S_a, S_o)            | 属性-对象独立的贡献         |
| +CLUB_all（完整）    | 所有三项 CLUB                  | 完整因果解耦的效果          |
| η 敏感性             | η ∈ {1e-4, 1e-3, 1e-2, 1e-1}  | 解耦强度的影响              |

---

## 8. 参数说明（新增超参数）

在 `.yml` 配置文件的 `train` 块中新增：

```yaml
causal:
  club_ao_weight: 1e-3       # CLUB(S_a, S_o) 权重
  club_az_weight: 1e-3       # CLUB(S_a, Z) 权重
  club_oz_weight: 1e-3       # CLUB(S_o, Z) 权重
  club_learn_weight: 1e-2    # CLUB 估计器学习损失权重
  club_hidden_dim: 512       # CLUB 估计器隐层维度
  use_z_branch: true         # 是否启用 Z 分支
```

---

## 9. 实验记录与对比指标

在三个数据集上记录 S / U / HM / AUC，重点关注：

- **U（Unseen accuracy）**：未见组合的识别准确率，因果解耦的直接受益项
- **HM（Harmonic Mean）**：兼顾 Seen 和 Unseen 的综合指标
- **AUC**：最综合的评估指标

预期效果：因果解耦后 U 和 HM 应有明显提升，S 可能略有下降（可接受的 trade-off）。


CUDA_VISIBLE_DEVICES=1 python train.py \
  --config configs/utzappos.yml \
  --dataset_path data/ut-zap50k