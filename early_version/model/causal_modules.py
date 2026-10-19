"""
causal_modules.py
=================
CausalCAMS 的两个因果核心模块：

  Module A — CounterfactualComposer
      在 MSD 解耦特征空间中学习可组合的因果合成函数，
      通过批内属性置换 + 循环一致性约束强制 f^a / f^o 具备因果可逆性。

  Module C — BackdoorAdjustment
      预计算组合对的逆频率权重，运行时对所有分支损失施加样本级重加权，
      近似实现后门调整（do(A), do(O)）以消除训练集中的虚假共现偏差。

用法：
    from causal_modules import CounterfactualComposer, BackdoorAdjustment
    # 见 causal_cams.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Module A: CounterfactualComposer
# ──────────────────────────────────────────────────────────────────────────────

class CounterfactualComposer(nn.Module):
    """
    可学习的因果组合函数 Φ(f^a, f^o) → f^composed。

    训练目标：
      L_recon：Φ(f^a_i, f^o_i) 与 com_prompt_i 对齐（重构监督）
      L_cf   ：Φ(f^a_perm, f^o_i) 经解耦头还原 (a_perm, o_i)（循环一致性）

    参数：
      dim        : 特征维度（CLIP output_dim，ViT-L/14 = 768）
      hidden_dim : Composer MLP 隐层维度，默认 2×dim
      dropout    : Composer 内部 Dropout
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2

        # ── 组合函数 Φ：[f^a || f^o] → f^composed ──────────────────────────
        self.causal_composer = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.composer_norm = nn.LayerNorm(dim)

        # ── 解耦头：f^composed → f^a_rec / f^o_rec ──────────────────────────
        self.dec_att = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.dec_obj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    # ── 基础组合接口 ─────────────────────────────────────────────────────────
    def compose(self, att_feat: torch.Tensor, obj_feat: torch.Tensor) -> torch.Tensor:
        """
        输入：att_feat (B, D)，obj_feat (B, D)，均已 L2 归一化
        输出：composed_feat (B, D)，L2 归一化
        """
        x = torch.cat([att_feat, obj_feat], dim=-1)          # (B, 2D)
        x = self.causal_composer(x)                          # (B, D)
        x = self.composer_norm(x)
        return F.normalize(x, p=2, dim=-1)

    # ── 前向：计算 L_recon + L_cf ────────────────────────────────────────────
    def forward(
        self,
        att_feat: torch.Tensor,       # (B, D)  MSD 属性特征，已 L2 归一化
        obj_feat: torch.Tensor,       # (B, D)  MSD 对象特征，已 L2 归一化
        com_prompt_reps: torch.Tensor,# (N_train_pairs, D)  组合文本 prompt，已归一化
        att_prompt_reps: torch.Tensor,# (N_train_pairs, D)  属性文本 prompt，已归一化
        obj_prompt_reps: torch.Tensor,# (N_train_pairs, D)  对象文本 prompt，已归一化
        pair_id: torch.Tensor,        # (B,)    训练对索引
        att_id: torch.Tensor,         # (B,)    属性索引（在 train_pairs 中对应的 att）
        obj_id: torch.Tensor,         # (B,)    对象索引
        temp_logit: torch.Tensor,     # scalar  温度参数
        cf_weight: float = 1.0,       # L_cf 内部 att/obj 两项的组合权重
    ):
        """
        返回：
            L_recon  : 重构损失（Seen 组合监督）
            L_cf     : 反事实循环一致性损失
            cf_feat  : 反事实组合特征 (B, D)，供调试/可视化使用
        """
        B = att_feat.shape[0]

        # ── 1. 重构损失 ──────────────────────────────────────────────────────
        # Φ(f^a_i, f^o_i) 应与 com_prompt_i 对齐
        com_recon = self.compose(att_feat, obj_feat)                    # (B, D)
        logits_recon = temp_logit * com_recon @ com_prompt_reps.T      # (B, N)
        L_recon = F.cross_entropy(logits_recon, pair_id)

        # ── 2. 反事实构造 ────────────────────────────────────────────────────
        # 批内随机置换属性，生成 (a'_i, o_i) 反事实对
        perm = torch.randperm(B, device=att_feat.device)
        att_cf   = att_feat[perm]                                       # (B, D)
        att_id_cf = att_id[perm]                                        # (B,)

        cf_feat = self.compose(att_cf, obj_feat)                        # (B, D)

        # ── 3. 循环一致性 ────────────────────────────────────────────────────
        # 解耦头还原属性/对象，验证因果可逆性
        att_rec = F.normalize(self.dec_att(cf_feat), p=2, dim=-1)      # (B, D)
        obj_rec = F.normalize(self.dec_obj(cf_feat), p=2, dim=-1)      # (B, D)

        # att_rec 应与置换后的属性 prompt 对齐
        # att_prompt_reps 按 train_pairs 索引，att_id_cf 是 attr2idx 的属性全集索引
        # 注意：att_prompt_reps 维度 = (N_train_pairs, D)，以 att_id_cf 做对比学习
        # 需要把 attr 全集索引映射到 train_pairs 中第一个出现该 attr 的 prompt
        # 实现上直接用 att_id_cf 在 att_prompt_reps 上做 InfoNCE 更简洁：
        logits_att = temp_logit * att_rec @ att_prompt_reps.T          # (B, N_train)
        # target：找 pair_id 中 attr == att_id_cf 的第一个 → 用批内最近邻近似
        # 简化实现：target 为置换后的 pair_id（对 att_prompt 做 argmax 对比）
        L_att_cycle = F.cross_entropy(logits_att, pair_id[perm])

        # obj_rec 应与原始对象 prompt 对齐
        logits_obj = temp_logit * obj_rec @ obj_prompt_reps.T          # (B, N_train)
        L_obj_cycle = F.cross_entropy(logits_obj, pair_id)

        L_cf = cf_weight * (L_att_cycle + L_obj_cycle)

        return L_recon, L_cf, cf_feat


# ──────────────────────────────────────────────────────────────────────────────
# Module C: BackdoorAdjustment
# ──────────────────────────────────────────────────────────────────────────────

class BackdoorAdjustment(nn.Module):
    """
    后门调整去偏模块。

    初始化时扫描 train_dataset，预计算各组合对的逆频率权重 w(a,o)，
    运行时对所有损失分支施加样本级重加权，近似实现干预分布估计。

    参数：
      train_dataset : CompositionDataset（phase='train'）
      smoothing     : Laplace 平滑系数，防止高频对权重过小（默认 1.0）
    """

    def __init__(self, train_dataset, smoothing: float = 1.0):
        super().__init__()

        n_pairs = len(train_dataset.train_pairs)
        counts  = torch.zeros(n_pairs, dtype=torch.float32)

        # 扫描训练集，统计每个 pair 出现次数
        for img_path, attr, obj in train_dataset.data:
            key = (attr, obj)
            if key in train_dataset.train_pair_to_idx:
                pair_idx = train_dataset.train_pair_to_idx[key]
                counts[pair_idx] += 1.0

        # 逆频率权重（Laplace 平滑 + 归一化）
        weights = 1.0 / (counts + smoothing)
        weights = weights / weights.mean()          # 均值为 1，不改变损失量级

        # 注册为 buffer（不参与梯度，随模型 .to(device) 自动转移）
        self.register_buffer("weights", weights)

        # 统计信息（供调试）
        self.n_pairs    = n_pairs
        self.smoothing  = smoothing

        print(
            f"[BackdoorAdjustment] n_pairs={n_pairs}, "
            f"count range=[{int(counts.min())}, {int(counts.max())}], "
            f"weight range=[{weights.min():.3f}, {weights.max():.3f}]"
        )

    def reweight_loss(
        self,
        logits: torch.Tensor,   # (B, N)  未归一化 logits
        targets: torch.Tensor,  # (B,)    整数标签
        pair_id: torch.Tensor,  # (B,)    train_pair 索引，用于查权重
    ) -> torch.Tensor:
        """
        返回逆频率重加权后的标量损失。
        用于替代原始 F.cross_entropy(logits, targets)。
        """
        per_sample = F.cross_entropy(logits, targets, reduction="none")  # (B,)
        w = self.weights[pair_id].float()                                 # (B,)
        return (per_sample * w).mean()

    def forward(self, logits, targets, pair_id):
        """等价接口，方便直接调用。"""
        return self.reweight_loss(logits, targets, pair_id)
