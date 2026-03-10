from __future__ import annotations

import torch
import torch.nn as nn


class ProprioQueryCrossAttentionEncoder(nn.Module):
    """Cross-attention encoder for proprioception–terrain reasoning (Eq. 1).

    - Query: proprioception (B, proprio_dim)
    - Key / Value: terrain scan sequence (B, T, scan_dim)
    - Output: latent (B, output_dim)
    """

    def __init__(
        self,
        scan_dim: int,
        proprio_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: int = 32,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        scan_conv1d_enabled: bool = False,
        scan_conv_channels: int = 64,
        scan_conv_kernel_size: int = 5,
        scan_conv_layers: int = 2,
        scan_conv_pool: str = "mean",
    ):
        super().__init__()
        if activation is None:
            activation = nn.ELU()

        self.scan_dim = int(scan_dim)
        self.proprio_dim = int(proprio_dim)
        self.d_model = int(d_model)

        self.scan_proj = nn.Sequential(
            nn.Linear(self.scan_dim, self.d_model),
            activation,
            nn.Linear(self.d_model, self.d_model),
        )

        # Optional Conv1D path for beam-wise feature extraction
        self.scan_conv1d_enabled = bool(scan_conv1d_enabled)
        self.scan_conv_pool = str(scan_conv_pool).strip().lower()
        if self.scan_conv1d_enabled:
            ch = int(max(4, scan_conv_channels))
            k = int(max(1, scan_conv_kernel_size))
            if k % 2 == 0:
                k += 1
            n_layers = int(max(1, scan_conv_layers))
            conv_layers = []
            in_ch = 1
            for _ in range(n_layers):
                conv_layers.append(nn.Conv1d(in_ch, ch, kernel_size=k, padding=k // 2))
                conv_layers.append(nn.GroupNorm(num_groups=min(8, ch), num_channels=ch))
                conv_layers.append(activation)
                in_ch = ch
            self.scan_conv = nn.Sequential(*conv_layers)
            self.scan_conv_out = nn.Sequential(
                nn.Linear(ch, self.d_model),
                activation,
                nn.Linear(self.d_model, self.d_model),
            )
        else:
            self.scan_conv = None
            self.scan_conv_out = None

        self.query_proj = nn.Sequential(
            nn.Linear(self.proprio_dim, self.d_model),
            activation,
            nn.Linear(self.d_model, self.d_model),
        )

        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.d_model,
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.ln1 = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(int(num_layers))]
        )
        self.ln2 = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(int(num_layers))]
        )
        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 2),
                    activation,
                    nn.Dropout(float(dropout)),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Dropout(float(dropout)),
                )
                for _ in range(int(num_layers))
            ]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, output_dim),
            activation,
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )

    def forward(
        self, proprio: torch.Tensor, scan_seq: torch.Tensor
    ) -> torch.Tensor:
        scan_seq = torch.nan_to_num(scan_seq, nan=0.0, posinf=1e6, neginf=-1e6)
        proprio = torch.nan_to_num(proprio, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scan_conv1d_enabled and self.scan_conv is not None:
            B, T, S = scan_seq.shape
            x = scan_seq.reshape(B * T, 1, S)
            feat = self.scan_conv(x)
            pooled = feat.amax(dim=-1) if self.scan_conv_pool == "max" else feat.mean(dim=-1)
            kv = self.scan_conv_out(pooled.reshape(B, T, -1))
        else:
            kv = self.scan_proj(scan_seq)

        kv = torch.nan_to_num(kv)
        q = torch.nan_to_num(self.query_proj(proprio)).unsqueeze(1)

        x = q
        for i, attn in enumerate(self.attn_layers):
            attn_out, _ = attn(query=x, key=kv, value=kv, need_weights=False)
            x = self.ln1[i](torch.nan_to_num(attn_out) + x)
            x = self.ln2[i](x + torch.nan_to_num(self.ff[i](x)))

        return self.output_proj(torch.nan_to_num(x.squeeze(1)))
