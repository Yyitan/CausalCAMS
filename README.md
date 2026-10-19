# CausalCAMS：CausalCAMS（SCM 结构因果模型 + CIB 因果解耦的 CAMS 改进，Step 1 验证性工作）

> 基于 [CAMS](https://github.com/ybyangjing/CAMS)（[arXiv:2511.16378](https://arxiv.org/abs/2511.16378)）的改进研究 · 个人研究项目

## 研究动机

CAMS 将图像特征分解为属性空间 S_a、对象空间 S_o 和组合空间，但解耦只依赖分类交叉熵监督，没有显式约束 S_a 与 S_o 的统计独立性——当模型学到"红色（属性）→ 苹果（对象）"这类共现偏见时，两个因子仍然相关，限制了对未见组合的泛化。本仓库在 CAMS 基础上引入结构因果模型（SCM）约束与 Causal Information Bottleneck（CIB）目标，用 CLUB 互信息上界估计显式最小化 S_a–S_o、S_a–Z、S_o–Z 之间的互信息。**当前为 Step 1 验证性工作：只加 SCM + CIB 因果解耦，不含扩散模型，实验数值尚未回填。**

## 方法

**SCM 设计**：引入属性因果因子 S_a（对应 CAMS `att` 分支）、对象因果因子 S_o（`obj` 分支），并新增非因果因子 Z（新 `tr_z`/`proj_z` 分支）用于"吸收"背景、光照等与标签无关的信息。约束 S_a ⊥ S_o、S_a ⊥ Z、S_o ⊥ Z；Z 不参与分类，只参与因果约束。推理流程与原始 CAMS 完全一致（不使用 Z 和 CLUB）。

**CIB 损失**：在 CAMS 原有四项 CE 损失之上，新增三项 CLUB 互信息上界正则与估计器自身的学习损失：

```
L_total = α_att·L_CE_attr + α_obj·L_CE_obj + α_com·L_CE_com + α_glb·L_CE_glb   (CAMS 原有)
        + η_ao·CLUB(S_a, S_o) + η_az·CLUB(S_a, Z) + η_oz·CLUB(S_o, Z)          (新增)
        + α_club·L_club_learn                                                   (新增)
```

CLUB（Cheng et al., ICML 2020）用可学习条件高斯 q_θ(y|x)=N(μ_θ(x), σ²_θ(x)) 给出互信息上界；实现采用 detach 策略在单次 backward 中同时训练估计器（对 detached 特征做 MLE）与编码器（最小化上界）。新增超参（yml `train` 块）：`club_ao_weight / club_az_weight / club_oz_weight`（默认 1e-3）、`club_learn_weight`（1e-2）、`club_hidden_dim`（512）。

| 文件 | 说明 |
|------|------|
| `CAMS/model/causal_modules.py` | 新增：`CLUBEstimator`（CLUB 互信息上界估计器） |
| `CAMS/clip_modules/gca.py` | 修改：新增 `tr_z`、`proj_z`，输出非因果因子 Z |
| `CAMS/model/cams.py` | 修改：实例化三个 CLUB 估计器，加入 CIB 损失 |
| `CAMS/model/configure_model.py` | 修改：因果模块参数加入优化器 |
| `CAMS/configs/*.yml` | 修改：mit-states / utzappos / cgqa 新增因果超参 |
| `CAMS/test.py` | 修改：解包 5 个输出 |
| `CAMS/CausalCAMS_Technical_Document.md` | 技术文档：SCM 设计、CIB 推导、CLUB 训练策略、消融设计 |
| `CAMS/CAMS_使用文档.md` | 面向使用者的环境/数据/训练/测试/消融指南 |
| `early_version/` | 早期版本：CounterfactualComposer（反事实组合 + 循环一致性）+ BackdoorAdjustment（逆频率重加权）方案，仅作记录，非当前主线 |

## 仓库内容

与系列其他仓库一致，本仓库只包含**新增模块、文档与配置**；对 CAMS 上游文件的修改以 `patches_from_upstream/*.diff` 记录（不包含上游原始代码，版权归原作者）。

```
CausalCAMS/
├── CAMS/                              # 新增文件
│   ├── model/causal_modules.py        # 新增：CLUBEstimator 等
│   ├── weight/init.txt
│   ├── CausalCAMS_Technical_Document.md
│   └── CAMS_使用文档.md
├── patches_from_upstream/             # 对 CAMS 上游文件的修改（git apply 即可应用）
│   ├── clip_modules__gca.py.diff      # 新增 Z 分支
│   ├── model__cams.py.diff            # CIB 损失
│   ├── model__configure_model.py.diff # 优化器参数组
│   ├── test.py.diff                   # 解包 5 个输出
│   └── configs__*.diff                # mit-states / utzappos / cgqa（含因果超参）
└── early_version/                     # 早期方案（Module A + Module C），仅存档
    ├── model/causal_cams.py           # CausalCAMS(CAMS) 继承版：反事实组合 + 后门调整
    ├── model/causal_modules.py
    └── configs/*-causal.yml
```

## 实验结果

本仓库为 Step 1 验证性工作，**尚未包含已完成的数值结果**（`CAMS_使用文档.md` 中的消融记录表为待填模板），因此这里不列数字，只给出实验设计：

- 消融设置：CAMS 基线 → +Z 分支（无 CLUB）→ +CLUB_ao → +CLUB_all（完整）→ η ∈ {1e-4, 1e-3, 1e-2, 1e-1} 敏感性扫描，在 MIT-States / UT-Zappos / C-GQA 上记录 S / U / HM / AUC；
- 预期方向（技术文档）：因果解耦主要作用于 U 与 HM，S 可能小幅下降（可接受的 trade-off）；
- 复现后请以实际训练输出为准，按使用文档第 7 节的命令逐项回填。

## 复现步骤

先获取 CAMS 官方代码，应用本仓库补丁，再放入新增文件：

```bash
# 0. 获取上游代码并应用本仓库的修改
git clone https://github.com/ybyangjing/CAMS.git && cd CAMS
git apply /path/to/CausalCAMS/patches_from_upstream/*.diff
# 再把本仓库 CAMS/ 下的新增文件（model/causal_modules.py、两个文档、weight/init.txt）复制到对应位置
cd ..

# 1. 准备环境（Python >= 3.8，PyTorch 2.0 + CUDA，ftfy regex tqdm pyyaml scipy numpy Pillow）
#    CLIP ViT-L/14 权重放在 ~/.cache/clip/ViT-L-14.pt（或自动下载）

# 2. 准备数据集（MIT-States / UT-Zappos / C-GQA，CZSL 标准 compositional-split-natural 切分，
#    目录结构见 CAMS/CAMS_使用文档.md 第 3 节）

# 3. 训练（注意：在 CAMS/ 的上一级目录运行；命令行覆盖因果超参暂不支持，直接改 yml）
CUDA_VISIBLE_DEVICES=0 python CAMS/train.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /path/to/ut-zap50k

# 4. 测试（closed-world / open-world）
CUDA_VISIBLE_DEVICES=0 python CAMS/test.py \
    --config CAMS/configs/utzappos.yml \
    --dataset_path /path/to/ut-zap50k \
    --load_model weight/val_best.pt \
    --open_world

# 5. 消融：把 yml 中 club_ao_weight / club_az_weight / club_oz_weight 置 0 或改为目标值，
#    按 CAMS_使用文档.md 第 7 节逐项对照
```

`early_version/` 中的早期方案仅供对照阅读，不在上述复现流程内。

## 引用

本仓库基于以下工作，如使用请同时引用：

```bibtex
@article{yang2025cams,
  title={CAMS: Towards Compositional Zero-Shot Learning via Gated Cross-Attention and Multi-Space Disentanglement},
  author={Yang, Pan and Deng, Cheng and Yang, Jing and Zhao, Han and Liu, Yun and Chen, Yuling and Ruan, Xiaoli and Chen, Yanping},
  journal={arXiv preprint arXiv:2511.16378},
  year={2025}
}
```

## License

本项目以 MIT License 发布（覆盖本仓库中用户新增/修改的部分；上游 CAMS 原始文件版权归原作者所有，且上游仓库未声明 LICENSE）。
