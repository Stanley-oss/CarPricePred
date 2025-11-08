import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Dual-Axis Transformer Block =====
class DualAxisTransformerBlock(nn.Module):
    def __init__(self, dim, channel, num_heads=4, dropout=0.1):
        super().__init__()

        # 注意：沿C维做注意力时，embed_dim应为D
        #       沿D维做注意力时，embed_dim应为C
        self.attn_c = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_d = nn.MultiheadAttention(embed_dim=channel, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, S, C, D]
        B, S, C, D = x.shape

        # ---- Attention along C ----
        # 每个样本（B,S）内部的C个通道互相注意
        x_c = x.reshape(B * S, C, D)               # [B*S, C, D]
        attn_c, _ = self.attn_c(x_c, x_c, x_c)     # 沿C维交互, embed_dim=D
        x_c = self.norm1(x_c + attn_c)

        # ---- Attention along D ----
        # 每个样本的D维特征之间交互
        x_d = x.permute(0, 1, 3, 2).reshape(B * S, D, C)  # [B*S, D, C]
        attn_d, _ = self.attn_d(x_d, x_d, x_d)            # 沿D维交互, embed_dim=C
        x_d = x_d + attn_d
        x_d = x_d.reshape(B, S, D, C).permute(0, 1, 3, 2) # 回到[B,S,C,D]
        x_d = self.norm2(x_d)

        # ---- Fusion ----
        x_out = (x_c.reshape(B, S, C, D) + x_d) / 2

        # ---- FFN ----
        x_out = x_out + self.ffn(x_out)
        x_out = self.norm3(x_out)
        return x_out


# ===== Full Model =====
class CarPriceTransformer(nn.Module):
    def __init__(self, input_dim, channel_dim, num_blocks=2, hidden_dim=16, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            DualAxisTransformerBlock(dim=hidden_dim, channel=channel_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
        self.reg_head = nn.Sequential(
            nn.Linear(channel_dim * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, xd):
        # xd: [B, S, C, D]
        xd = self.input_proj(xd)  # [B, S, C, hidden_dim]
        for blk in self.blocks:
            xd = blk(xd)
        print(xd.shape)
        xd_flat = xd.reshape(B, S, C * xd.shape[-1])
        out = self.reg_head(xd_flat)
        return out.squeeze(-1)


# ====== 测试 ======
if __name__ == "__main__":
    B, S, C, D = 8, 10, 8, 1
    x = torch.randn(B, S, C, D)
    model = CarPriceTransformer(input_dim=D, channel_dim=C, num_blocks=2, hidden_dim=16, num_heads=4)
    print(model)
    y = model(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
