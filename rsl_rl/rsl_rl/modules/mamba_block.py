import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaBlock(nn.Module):
    """Selective State Space block following the Mamba formulation:

        h_t = A_t h_{t-1} + B_t x_t,   y_t = C_t h_t        (Eq. 3)

    (A_t, B_t, C_t) are data-dependent: A is discretized via a learned
    per-step Δ_t, while B_t and C_t are projected from the input.

    Input:  x (B, F, D), optional proprio (B, P).
    Output: (B, F, D).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dropout: float = 0.0,
        proprio_dim: int = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        self.in_proj = nn.Linear(self.d_model, self.d_model * 2)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.dt_proj = nn.Linear(self.d_model, self.d_model)
        self.B_proj = nn.Linear(self.d_model, self.d_state)
        self.C_proj = nn.Linear(self.d_model, self.d_state)

        A_log = torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(-A_log.unsqueeze(0).expand(self.d_model, -1).clone())

        if proprio_dim is not None:
            self.proprio_proj = nn.Linear(proprio_dim, self.d_model * 2)
        else:
            self.proprio_proj = None

        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        proprio: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"SimpleMambaBlock expects (B,F,{self.d_model}), got {tuple(x.shape)}"
            )
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        if self.proprio_proj is not None and proprio is not None:
            pg, pv = self.proprio_proj(proprio).chunk(2, dim=-1)
            x_branch = x_branch + pg.unsqueeze(1)
            z = z + pv.unsqueeze(1)

        x_branch = F.silu(x_branch)

        dt = F.softplus(self.dt_proj(x_branch))   # (B, T, D)
        B_t = self.B_proj(x_branch)                # (B, T, N)
        C_t = self.C_proj(x_branch)                # (B, T, N)

        A = -torch.exp(self.A_log)                  # (D, N), negative for decay

        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)                       # (B, D, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))               # (B, D, N)
            dB = dt_t * B_t[:, t, :].unsqueeze(1)                  # (B, D, N)
            h = A_bar * h + dB * x_branch[:, t, :].unsqueeze(-1)   # (B, D, N)
            y_t = (h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1)      # (B, D)
            ys.append(y_t)

        y_seq = torch.stack(ys, dim=1)
        out = y_seq * torch.sigmoid(z)
        out = self.out_proj(out)
        out = self.dropout(out)
        return self.norm(x + out)
