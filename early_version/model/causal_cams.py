"""
causal_cams.py
==============
CausalCAMS：在 CAMS 基础上融合因果推断的组合零样本学习模型。

新增两个因果模块（详见 causal_modules.py）：
  - CounterfactualComposer (Module A)：反事实组合 + 循环一致性约束
  - BackdoorAdjustment     (Module C)：逆频率重加权，近似后门调整

所有原始 CAMS 组件（CLIP encoder、GCA、MSD、CSP）保持不变。
CausalCAMS 完全继承 CAMS，仅重写 forward() 和 __init__()。

使用方式（configure_model.py 会自动选择）：
    model = CausalCAMS(args, train_dataset)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cams import CAMS
from model.causal_modules import CounterfactualComposer, BackdoorAdjustment


class CausalCAMS(CAMS):
    """
    CausalCAMS = CAMS + Module A (CounterfactualComposer)
                      + Module C (BackdoorAdjustment)

    额外 args 字段（通过 yml config 传入）：
        use_causal           (bool, default True)   总开关
        causal_recon_weight  (float, default 0.5)   L_recon 权重 λ₁
        causal_cf_weight     (float, default 0.3)   L_cf    权重 λ₂
        cf_hidden_dim        (int,   default None)  Composer 隐层维度，None = 2×D
        cf_dropout           (float, default 0.1)  Composer Dropout
        backdoor_smoothing   (float, default 1.0)  逆频率 Laplace 平滑
    """

    def __init__(self, args, dataset):
        super().__init__(args, dataset)          # 完整初始化原始 CAMS

        self.use_causal = getattr(args, "use_causal", True)

        if not self.use_causal:
            print("[CausalCAMS] use_causal=False, running as standard CAMS.")
            return

        # ── 读取因果超参 ──────────────────────────────────────────────────────
        self.causal_recon_weight = getattr(args, "causal_recon_weight", 0.5)
        self.causal_cf_weight    = getattr(args, "causal_cf_weight",    0.3)

        # ── 推断 CLIP 特征维度（output_dim）─────────────────────────────────
        # CLIP ViT-L/14 output_dim = 768；ViT-B/16 = 512
        dim = self.clip_model.visual.output_dim
        hidden_dim = getattr(args, "cf_hidden_dim", None)    # None → 2×dim
        dropout    = getattr(args, "cf_dropout", 0.1)
        smoothing  = getattr(args, "backdoor_smoothing", 1.0)

        # ── Module A: CounterfactualComposer ─────────────────────────────────
        self.causal_composer = CounterfactualComposer(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ── Module C: BackdoorAdjustment ─────────────────────────────────────
        self.backdoor = BackdoorAdjustment(
            train_dataset=dataset,
            smoothing=smoothing,
        )

        print(
            f"[CausalCAMS] dim={dim}, "
            f"λ₁(recon)={self.causal_recon_weight}, "
            f"λ₂(cf)={self.causal_cf_weight}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, data):
        img, att_id, obj_id, pair_id = data[0], data[1], data[2], data[3]

        # ── CAMS 原始四路特征提取（完全不变）────────────────────────────────
        att, obj, com, glb = self.image_encoder(img)
        att_sem = F.normalize(att, p=2, dim=-1)
        obj_sem = F.normalize(obj, p=2, dim=-1)
        com_sem = F.normalize(com, p=2, dim=-1)
        glb_sem = F.normalize(glb, p=2, dim=-1)

        # ── 文本 prompt 特征（与原始 CAMS 完全相同）─────────────────────────
        token_com, token_att, token_obj = self.csp.construct_token_tensors(self.train_pairs)
        com_prompt, att_prompt, obj_prompt = self.get_text_features(
            token_com, token_att, token_obj
        )

        # ── 四路 logits（与原始 CAMS 完全相同）──────────────────────────────
        att_branch = self.temp_logit * att_sem @ att_prompt.T   # (B, N_train)
        obj_branch = self.temp_logit * obj_sem @ obj_prompt.T   # (B, N_train)
        com_branch = self.temp_logit * com_sem @ com_prompt.T   # (B, N_train)
        glb_branch = self.temp_logit * glb_sem @ com_prompt.T   # (B, N_train)

        # ── [Module C] 后门调整重加权损失 ────────────────────────────────────
        if self.use_causal:
            att_loss = self.backdoor.reweight_loss(att_branch, att_id, pair_id)
            obj_loss = self.backdoor.reweight_loss(obj_branch, obj_id, pair_id)
            com_loss = self.backdoor.reweight_loss(com_branch, pair_id, pair_id)
            glb_loss = self.backdoor.reweight_loss(glb_branch, pair_id, pair_id)
        else:
            # 退化为标准 CAMS
            att_loss = F.cross_entropy(att_branch, att_id)
            obj_loss = F.cross_entropy(obj_branch, obj_id)
            com_loss = F.cross_entropy(com_branch, pair_id)
            glb_loss = F.cross_entropy(glb_branch, pair_id)

        # ── CAMS 四路加权总损失 ───────────────────────────────────────────────
        cams_loss = (
            self.args.att_loss_weight * att_loss +
            self.args.obj_loss_weight * obj_loss +
            self.args.com_loss_weight * com_loss +
            self.args.glb_loss_weight * glb_loss
        )

        # ── [Module A] 反事实组合损失 ─────────────────────────────────────────
        if self.use_causal:
            L_recon, L_cf, _ = self.causal_composer(
                att_feat       = att_sem,
                obj_feat       = obj_sem,
                com_prompt_reps= com_prompt,
                att_prompt_reps= att_prompt,
                obj_prompt_reps= obj_prompt,
                pair_id        = pair_id,
                att_id         = att_id,
                obj_id         = obj_id,
                temp_logit     = self.temp_logit,
                cf_weight      = 1.0,
            )

            total_loss = (
                cams_loss
                + self.causal_recon_weight * L_recon
                + self.causal_cf_weight    * L_cf
            )
        else:
            total_loss = cams_loss

        return total_loss
