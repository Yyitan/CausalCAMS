# CausalCAMS 使用文档

## 目录
1. [环境配置](#1-环境配置)
2. [目录结构说明](#2-目录结构说明)
3. [数据集准备](#3-数据集准备)
4. [CLIP 权重下载](#4-clip-权重下载)
5. [训练命令](#5-训练命令)
6. [测试命令](#6-测试命令)
7. [消融实验命令](#7-消融实验命令)
8. [配置文件说明](#8-配置文件说明)
9. [常见报错与解决](#9-常见报错与解决)

---

## 1. 环境配置

### 1.1 Python 环境要求

```bash
Python >= 3.8
CUDA   >= 11.3
```

### 1.2 安装依赖

```bash
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install ftfy regex tqdm pyyaml scipy numpy Pillow
```

> 如果你的 CUDA 版本不是 11.8，去 https://pytorch.org/get-started/locally 选对应版本的安装命令

### 1.3 验证 GPU 可用

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

输出 `True` + GPU 名称即为正常。

---

## 2. 目录结构说明

解压代码后，整体结构如下：

```
CAMS/                              ← 你运行命令时所在的根目录
├── clip_modules/
│   ├── clip.py
│   ├── gca.py                     ← 已修改（新增 Z 分支）
│   ├── model.py
│   ├── text_encoder.py
│   └── simple_tokenizer.py
├── model/
│   ├── cams.py                    ← 已修改（加入 CLUB 损失）
│   ├── causal_modules.py          ← 新增（CLUB 估计器）
│   ├── configure_model.py         ← 已修改（新参数加入优化器）
│   └── csp.py
├── configs/
│   ├── mit-states.yml             ← 已更新（新增因果超参）
│   ├── utzappos.yml               ← 已更新（新增因果超参）
│   └── cgqa.yml                   ← 已更新（新增因果超参）
├── feasibilities/
│   ├── feasibility_mit.pt
│   ├── feasibility_zp.pt
│   └── feasibility_cgqa.pt
├── weight/                        ← 训练时自动保存模型到这里
├── train.py
├── test.py
├── dataset.py
├── flags.py
└── utils.py
```

**所有命令都必须在 `CAMS/` 目录的上一级运行**，即包含 `CAMS/` 文件夹的那一层。

```
你的工作目录/
└── CAMS/         ← 这个文件夹
    ├── train.py
    └── ...
```

---

## 3. 数据集准备

需要三个数据集，**每个数据集的文件夹结构必须完全一致**，否则代码报错。

### 3.1 MIT-States

**下载地址**：http://web.mit.edu/phillipi/Public/states_and_transformations/index.html

下载后整理成以下结构：

```
mit-states/
├── images/
│   ├── adj noun/         ← 例如 "sliced apple/"
│   │   ├── xxx.jpg
│   │   └── ...
│   └── ...
├── compositional-split-natural/
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
└── metadata_compositional-split-natural.t7
```

> `metadata_*.t7` 和 `compositional-split-natural/` 文件夹来自 CZSL 社区的标准 split，
> 可从这里获取：https://github.com/ExplainableML/czsl

### 3.2 UT-Zappos50K

**下载地址**：https://vision.cs.utexas.edu/projects/finegrained/utzap50k/

整理后结构：

```
ut-zap50k/
├── images/
│   ├── Boots/
│   │   ├── Ankle/
│   │   │   └── xxx.jpg
│   │   └── ...
│   └── ...
├── compositional-split-natural/
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
└── metadata_compositional-split-natural.t7
```

### 3.3 C-GQA

**下载地址**：https://github.com/ExplainableML/czsl（论文仓库提供下载脚本）

整理后结构：

```
cgqa/
├── images/
│   └── ...
├── compositional-split-natural/
│   ├── train_pairs.txt
│   ├── val_pairs.txt
│   └── test_pairs.txt
└── metadata_compositional-split-natural.t7
```

### 3.4 将数据集路径填入配置文件

打开对应的 `.yml` 文件，**修改 `dataset_path` 这一行**（如果配置文件里没有，就在命令行里指定，见第5节）。

---

## 4. CLIP 权重下载

代码会自动下载 CLIP 权重，但需要科学上网。如果服务器没有网络，用以下方法手动处理：

### 方法一：自动下载（有网络时）

代码第一次运行时会自动下载到 `~/.cache/clip/` 目录，无需额外操作。

### 方法二：手动下载（服务器无网络时）

**在本地**下载：
```
ViT-L/14 下载地址：
https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd5d/ViT-L-14.pt
```

下载完成后传到服务器，放到指定位置：
```bash
mkdir -p ~/.cache/clip
cp ViT-L-14.pt ~/.cache/clip/ViT-L-14.pt
```

---

## 5. 训练命令

**所有命令都在包含 `CAMS/` 文件夹的那一级目录下运行。**

### 5.1 MIT-States 训练

```bash
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states
```

### 5.2 UT-Zappos 训练

```bash
python CAMS/train.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /你的数据集路径/ut-zap50k
```

### 5.3 C-GQA 训练

```bash
python CAMS/train.py \
    --config CAMS/configs/cgqa.yml \
    --dataset_path /你的数据集路径/cgqa
```

### 5.4 指定 GPU 运行

```bash
# 只用第 0 块 GPU
CUDA_VISIBLE_DEVICES=0 python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states

# 只用第 1 块 GPU
CUDA_VISIBLE_DEVICES=1 python CAMS/train.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /你的数据集路径/ut-zap50k
```

### 5.5 后台运行并保存日志（服务器上推荐）

```bash
CUDA_VISIBLE_DEVICES=0 nohup python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    > logs/mit_causal.log 2>&1 &

# 查看训练进度
tail -f logs/mit_causal.log
```

先建 logs 目录：
```bash
mkdir -p logs
```

### 5.6 训练输出说明

训练过程中你会看到：

```
[configure_model] Trainable: 123,456,789 / 400,000,000 (30.86%)
epoch   1: 100%|████████| 843/843 [03:12<00:00, train loss: 2.4531]
epoch 1 train loss 2.4531
Evaluating val dataset:
best_seen  0.5123| best_unseen  0.4231| best_hm  0.3987| AUC  0.2134| ...
```

每个 epoch 结束会评估验证集，`AUC` 是最重要的指标，关注它是否在提升。

### 5.7 模型保存位置

训练过程中自动保存到 `CAMS/weight/`：
- `val_best.pt` ← **验证集上最好的模型**，最终测试用这个
- `epoch_N.pt` ← 每 5 个 epoch 保存一次（可以用 `--save_every_n` 改频率）
- `final_model.pt` ← 最后一个 epoch 的模型

---

## 6. 测试命令

### 6.1 Closed-World 测试

```bash
# MIT-States
python CAMS/test.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --load_model CAMS/weight/val_best.pt

# UT-Zappos
python CAMS/test.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /你的数据集路径/ut-zap50k \
    --load_model CAMS/weight/val_best.pt

# C-GQA
python CAMS/test.py \
    --config CAMS/configs/cgqa.yml \
    --dataset_path /你的数据集路径/cgqa \
    --load_model CAMS/weight/val_best.pt
```

### 6.2 Open-World 测试

```bash
# MIT-States Open-World
python CAMS/test.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --load_model CAMS/weight/val_best.pt \
    --open_world

# UT-Zappos Open-World（配置文件里默认已开 open_world: True）
python CAMS/test.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /你的数据集路径/ut-zap50k \
    --load_model CAMS/weight/val_best.pt
```

### 6.3 测试输出说明

```
best_seen  0.5312| best_unseen  0.4456| best_hm  0.4123| AUC  0.2456| attr_acc  0.7234| obj_acc  0.8123|
```

| 指标 | 含义 |
|---|---|
| best_seen (S) | 见过组合的最高准确率 |
| best_unseen (U) | 未见组合的最高准确率 |
| best_hm (HM) | S 和 U 的调和平均，越高越好 |
| AUC | 综合评估指标，**最重要**，越高越好 |
| attr_acc | 属性单独识别准确率 |
| obj_acc | 对象单独识别准确率 |

---

## 7. 消融实验命令

这部分专门用于验证因果解耦的效果，按顺序跑，对比 AUC 变化。

### 实验一：原始 CAMS 基线（无因果约束）

把三个 club weight 全设为 0，只保留 Z 分支：

```bash
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --club_ao_weight 0 \
    --club_az_weight 0 \
    --club_oz_weight 0
```

> 注意：命令行里的参数会覆盖配置文件里的值。

### 实验二：只加属性-对象解耦

```bash
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --club_ao_weight 0.001 \
    --club_az_weight 0 \
    --club_oz_weight 0
```

### 实验三：完整 CausalCAMS（默认配置）

```bash
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states
# 直接用配置文件默认值即可，三个 weight 都是 0.001
```

### 实验四：η 敏感性扫描

```bash
# η = 1e-4（弱解耦）
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --club_ao_weight 0.0001 \
    --club_az_weight 0.0001 \
    --club_oz_weight 0.0001

# η = 1e-3（默认，中等）
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states

# η = 1e-2（强解耦）
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /你的数据集路径/mit-states \
    --club_ao_weight 0.01 \
    --club_az_weight 0.01 \
    --club_oz_weight 0.01
```

### 让命令行参数生效的前提

`flags.py` 目前没有这些因果参数的 `add_argument`，所以**命令行覆盖暂时不支持**，用以下两种方法替代：

**方法 A（推荐）：直接改 yml 文件**

```yaml
# CAMS/configs/mit-states.yml 里直接改这几行
club_ao_weight: 0.0001   # 改成你想要的值
club_az_weight: 0.0001
club_oz_weight: 0.0001
```

**方法 B：为每个实验建一个配置文件**

```bash
cp CAMS/configs/mit-states.yml CAMS/configs/mit-states-ablation1.yml
# 然后编辑 mit-states-ablation1.yml 里的 club_*_weight
python CAMS/train.py \
    --config CAMS/configs/mit-states-ablation1.yml \
    --dataset_path /你的数据集路径/mit-states
```

### 消融实验结果记录表

| 实验 | club_ao | club_az | club_oz | S | U | HM | AUC |
|---|---|---|---|---|---|---|---|
| CAMS 基线 | 0 | 0 | 0 | | | | |
| +Z分支 only | 0 | 0 | 0 | | | | |
| +CLUB_ao | 0.001 | 0 | 0 | | | | |
| +CLUB_all (η=1e-4) | 0.0001 | 0.0001 | 0.0001 | | | | |
| +CLUB_all (η=1e-3) | 0.001 | 0.001 | 0.001 | | | | |
| +CLUB_all (η=1e-2) | 0.01 | 0.01 | 0.01 | | | | |

---

## 8. 配置文件说明

以 `mit-states.yml` 为例，所有字段解释：

```yaml
model:
  model_name: CAMS
  clip_model: "ViT-L/14"     # CLIP 主干，不要改
  num_units: 32              # Latent Units 数量，论文用 32
  m_layers: 8                # 最后 M 层用 LoRA 微调（MIT=8, ZAP=9, CGQA=12）
  adapter_dim: 128           # LoRA 瓶颈维度

train:
  lr: 0.0001                 # 学习率
  train_batch_size: 64       # 批大小，显存不够就改成 32
  epochs: 15                 # 训练轮数
  step_size: 5               # 学习率每 5 epoch 乘以 gamma
  gamma: 0.5                 # 学习率衰减系数
  val_metric: AUC            # 用 AUC 选最好的模型保存
  att_loss_weight: 0.5       # 属性分支损失权重
  obj_loss_weight: 0.5       # 对象分支损失权重
  com_loss_weight: 1         # 组合分支损失权重
  glb_loss_weight: 1         # 全局分支损失权重

  # ── 因果解耦超参（CausalCAMS 新增）──────────────────
  club_ao_weight: 0.001      # CLUB(S_a, S_o) 权重，控制属性-对象解耦强度
  club_az_weight: 0.001      # CLUB(S_a, Z)   权重，控制属性-背景解耦强度
  club_oz_weight: 0.001      # CLUB(S_o, Z)   权重，控制对象-背景解耦强度
  club_learn_weight: 0.01    # CLUB 估计器自身的学习率权重，保持比上面大一量级
  club_hidden_dim: 512       # CLUB 估计器 MLP 的隐层维度

test:
  beta: 0.85                 # 推理时全局分支的混合权重
  threshold: 0.3625          # Open-world 下的可行性过滤阈值
  feasibility: CAMS/feasibilities/feasibility_mit.pt  # 可行性文件路径
  open_world: false          # false=Closed-world, true=Open-world
```

---

## 9. 常见报错与解决

### 报错 1：`ModuleNotFoundError: No module named 'clip_modules'`

**原因**：没有在正确目录运行命令。

**解决**：确保你在 `CAMS/` 的**上一级目录**运行，且命令开头是 `python CAMS/train.py`，不是 `python train.py`。

```bash
# 错误写法（在 CAMS/ 内部运行）
cd CAMS
python train.py ...           # ❌

# 正确写法（在 CAMS/ 外面运行）
cd ..                         # 退到 CAMS 的上层
python CAMS/train.py ...      # ✅
```

### 报错 2：`FileNotFoundError: metadata_compositional-split-natural.t7`

**原因**：数据集路径不对，或 metadata 文件缺失。

**解决**：
1. 检查 `--dataset_path` 指向的路径下是否有 `metadata_compositional-split-natural.t7`
2. 该文件从 CZSL 标准 split 获取：https://github.com/ExplainableML/czsl

### 报错 3：`CUDA out of memory`

**原因**：显存不足。

**解决**：在配置文件里把 `train_batch_size` 从 64 改为 32：
```yaml
train_batch_size: 32
```

### 报错 4：`KeyError: 'club_ao_weight'`

**原因**：配置文件里缺少因果超参字段。

**解决**：确认使用的是 CausalCAMS 更新后的配置文件，或手动在 yml 的 `train:` 块下添加：
```yaml
club_ao_weight: 0.001
club_az_weight: 0.001
club_oz_weight: 0.001
club_learn_weight: 0.01
club_hidden_dim: 512
```

### 报错 5：`RuntimeError: Expected all tensors to be on the same device`

**原因**：数据或模型有部分没有移到 GPU。

**解决**：确认你的机器有 GPU 且 PyTorch 能识别到：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
如果输出 `False`，检查 CUDA 驱动和 PyTorch 版本是否匹配。

### 报错 6：CLIP 模型下载失败 / 超时

**解决**：手动下载放到 `~/.cache/clip/ViT-L-14.pt`，参考第 4 节。

---

## 快速上手（TL;DR）

如果不想细看，按这 4 步操作：

```bash
# Step 1：安装依赖
pip install torch torchvision ftfy regex tqdm pyyaml scipy numpy Pillow

# Step 2：把数据集放好，记住路径，比如 /data/mit-states

# Step 3：在 CAMS/ 外面运行训练
python CAMS/train.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /data/mit-states

# Step 4：训练结束后运行测试
python CAMS/test.py \
    --config CAMS/configs/mit-states.yml \
    --dataset_path /data/mit-states \
    --load_model CAMS/weight/val_best.pt
```

关注终端输出里的 `AUC` 值，与原始 CAMS 论文数据（MIT-States: 24.2）对比，高于它说明因果解耦有效。
