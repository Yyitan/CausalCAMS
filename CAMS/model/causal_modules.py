"""
causal_modules.py
─────────────────
CausalCAMS 新增模块：因果信息瓶颈（CIB）所需的 CLUB 互信息上界估计器。

CLUB（Contrastive Log-ratio Upper Bound）：
    I(X;Y) ≤ E_{p(x,y)}[log q_θ(y|x)] - E_{p(x)}E_{p(y)}[log q_θ(y|x)]

其中 q_θ(y|x) = N(μ_θ(x), σ²_θ(x)) 是一个可学习的条件高斯分布。

参考：
    Cheng et al., "CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information",
    ICML 2020.
"""

import torch
import torch.nn as nn


class CLUBEstimator(nn.Module):
    """
    CLUB 互信息上界估计器。

    给定两个特征向量 x 和 y，估计 I(X;Y) 的上界，用于最小化两者之间的
    统计相关性（解耦）。

    用法：
        estimator = CLUBEstimator(x_dim=768, y_dim=768, hidden_dim=512)

        # 训练估计器本身（使上界更紧）
        learn_loss = estimator.learning_loss(x.detach(), y.detach())

        # 为主模型提供解耦损失（使 MI 更小）
        club_bound = estimator(x, y)
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.y_dim = y_dim
        # 条件高斯分布的参数网络：x → (μ, log σ²)
        self.predictor = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim * 2),  # 前半为 μ，后半为 log σ²
        )

    def get_mu_logvar(self, x: torch.Tensor):
        out = self.predictor(x)
        mu, log_var = out.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -10.0, 2.0)
        return mu, log_var

    def _log_prob(self, mu, log_var, y):
        return -0.5 * (((y - mu) ** 2) / log_var.exp() + log_var).sum(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算 CLUB 互信息上界（用于训练编码器，最小化 MI）。
        Args:
            x, y: [B, dim]，不要 detach，梯度需要流向编码器
        Returns:
            scalar，CLUB 上界估计值
        """
        mu, log_var = self.get_mu_logvar(x)
        # 正样本项
        pos = self._log_prob(mu, log_var, y).mean()
        # 负样本项（所有交叉对）
        mu_exp      = mu.unsqueeze(1)
        log_var_exp = log_var.unsqueeze(1)
        y_exp       = y.unsqueeze(0)
        neg = self._log_prob(mu_exp, log_var_exp, y_exp).mean()
        return pos - neg

    def learning_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        训练 CLUB 估计器的损失（负对数似然）。
        调用时请传入 detached 的 x, y，使梯度只流向估计器网络。
        """
        mu, log_var = self.get_mu_logvar(x)
        return -self._log_prob(mu, log_var, y).mean()


class CausalIBLoss(nn.Module):
    """
    Causal Information Bottleneck 损失模块。

    整合三个 CLUB 估计器（ao / az / oz），统一计算：
        L_causal = η_ao · CLUB(S_a, S_o)
                 + η_az · CLUB(S_a, Z)
                 + η_oz · CLUB(S_o, Z)
                 + α_club · [learn(S_a,S_o) + learn(S_a,Z) + learn(S_o,Z)]
    """

    def __init__(
        self,
        feat_dim: int = 768,
        hidden_dim: int = 512,
        club_ao_weight: float = 1e-3,
        club_az_weight: float = 1e-3,
        club_oz_weight: float = 1e-3,
        club_learn_weight: float = 1e-2,
    ):
        super().__init__()
        self.club_ao_weight    = club_ao_weight
        self.club_az_weight    = club_az_weight
        self.club_oz_weight    = club_oz_weight
        self.club_learn_weight = club_learn_weight

        # 三个独立的 CLUB 估计器
        self.club_ao = CLUBEstimator(feat_dim, feat_dim, hidden_dim)  # S_o → S_a
        self.club_az = CLUBEstimator(feat_dim, feat_dim, hidden_dim)  # Z   → S_a
        self.club_oz = CLUBEstimator(feat_dim, feat_dim, hidden_dim)  # Z   → S_o

    def forward(self, s_a, s_o, z_nc):
        """
        Args:
            s_a:  属性因果因子 [B, feat_dim]（未归一化）
            s_o:  对象因果因子 [B, feat_dim]（未归一化）
            z_nc: 非因果因子   [B, feat_dim]（未归一化）
        Returns:
            total_causal_loss: scalar
            loss_dict: dict，各项损失明细
        """
        # ── 1. CLUB 上界（训练编码器，梯度流向编码器）──────────────────────
        bound_ao = self.club_ao(s_a, s_o)
        bound_az = self.club_az(s_a, z_nc)
        bound_oz = self.club_oz(s_o, z_nc)

        # ── 2. CLUB 估计器学习损失（用 detached 特征，只传梯度给估计器）──────
        s_a_d = s_a.detach()
        s_o_d = s_o.detach()
        z_d   = z_nc.detach()
        learn_ao    = self.club_ao.learning_loss(s_o_d, s_a_d)
        learn_az    = self.club_az.learning_loss(z_d,   s_a_d)
        learn_oz    = self.club_oz.learning_loss(z_d,   s_o_d)
        learn_total = learn_ao + learn_az + learn_oz

        # ── 3. 加权求和 ──────────────────────────────────────────────────────
        total_causal_loss = (
            self.club_ao_weight    * bound_ao
            + self.club_az_weight  * bound_az
            + self.club_oz_weight  * bound_oz
            + self.club_learn_weight * learn_total
        )

        loss_dict = {
            "club_ao":      bound_ao.item(),
            "club_az":      bound_az.item(),
            "club_oz":      bound_oz.item(),
            "club_learn":   learn_total.item(),
            "causal_total": total_causal_loss.item(),
        }
        return total_causal_loss, loss_dict
