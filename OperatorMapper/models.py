"""模型定义模块 - 包含算子映射器和相关组件"""

import torch
import torch.nn as nn

class OperatorMapper(nn.Module):
    """算子映射器的三个关键组件解析"""
    
    def __init__(self, spec_shape=(129, 32), rank=64, latent_dim=1024):
        super().__init__()
        self.F, self.T = spec_shape
        self.in_dim = self.F * self.T
        self.rank = rank
        self.latent_dim = latent_dim
        
        # 【核心1】低秩算子分解 - 这是论文的关键创新
        # 传统方法：直接学习 W: in_dim -> latent_dim (参数量巨大)
        # 低秩方法：W = B @ A, 其中 B: in_dim->rank, A: rank->latent_dim
        # 优势：参数量从 in_dim*latent_dim 降到 (in_dim+latent_dim)*rank
        self.B = nn.Parameter(torch.randn(self.in_dim, rank) * 0.01)
        self.A = nn.Parameter(torch.randn(rank, latent_dim) * 0.01)
        
        # 非线性激活
        self.act = nn.GELU()
        
        # 解码器：潜在空间 -> 64x64 图像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8*8*128),
            nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 64x64
        )

    def forward(self, spec):
        """前向传播的三阶段流程"""
        Bsz = spec.shape[0]
        
        # 【阶段1】谱系数展平 - 将2D频谱转为1D向量
        x = spec.reshape(Bsz, -1)  # [B, F*T] -> [B, in_dim]
        
        # 【阶段2】低秩算子映射 - 论文核心
        # 这里实现了受控的线性变换：x -> z
        # 分两步：x @ B -> 中间表示(rank维) -> @ A -> 潜在表示
        z = x @ self.B          # [B, in_dim] @ [in_dim, rank] = [B, rank]
        z = z @ self.A          # [B, rank] @ [rank, latent] = [B, latent_dim]
        z = self.act(z)         # 轻微非线性，保持可解释性
        
        # 【阶段3】图像重建 - 从潜在空间解码到高分辨率图像
        out = self.decoder(z)   # [B, latent] -> [B, 1, 64, 64]
        return out.squeeze(1), z

    def get_operator_regularization(self):
        """算子正则化 - 控制模型复杂度和泛化能力"""
        # Frobenius范数惩罚，鼓励低秩结构
        # 这相当于对 ||A||_F^2 + ||B||_F^2 的惩罚
        # 间接控制了 ||A@B||_* (核范数) 的上界
        return self.A.pow(2).sum() + self.B.pow(2).sum()
