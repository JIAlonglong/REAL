import torch
import torch.nn as nn

from rsl_rl.modules.mamba_block import SimpleMambaBlock


# ---------------------------------------------------------------------------
# FiLM2d  —  Feature-wise Linear Modulation (Eq. 2)
# ---------------------------------------------------------------------------

class FiLM2d(nn.Module):
    """Channel-wise affine modulation: γ(p) ⊙ x + β(p)."""

    def __init__(self, prop_dim: int, channels: int,
                 gamma_scale: float = 0.3, beta_scale: float = 0.3,
                 activation: nn.Module = nn.ELU()):
        super().__init__()
        self.prop_dim = int(prop_dim)
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.prop_dim)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(self.prop_dim, self.channels), activation,
            nn.Linear(self.channels, self.channels), nn.Tanh(),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(self.prop_dim, self.channels), activation,
            nn.Linear(self.channels, self.channels), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        Bx, Bp = x.shape[0], p.shape[0]
        if Bp != Bx and Bx % Bp == 0:
            p = p.repeat_interleave(Bx // Bp, dim=0)
        p = self.norm(p)
        gamma = 1.0 + self.gamma_scale * self.gamma_mlp(p).view(Bx, self.channels, 1, 1)
        beta = self.beta_scale * self.beta_mlp(p).view(Bx, self.channels, 1, 1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# DepthOnlyFCBackbone58x87  —  CNN backbone with FiLM
# ---------------------------------------------------------------------------

class DepthOnlyFCBackbone58x87(nn.Module):
    """CNN backbone (58×87 depth) with optional FiLM modulation (Eq. 2).

    Two FiLM layers are inserted after conv1 and conv2 respectively.
    Activation is controlled by ``film_spatial_enabled`` (set by the
    enclosing ``RecurrentDepthBackbone``).
    """

    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim,
                 output_activation=None, num_frames=1, cnn_channels=None):
        super().__init__()
        self.num_frames = num_frames
        self.prop_dim = int(prop_dim)
        activation = nn.ELU()
        c1, c2 = cnn_channels if cnn_channels is not None else (32, 64)
        self.c1, self.c2 = int(c1), int(c2)

        self.conv1 = nn.Conv2d(self.num_frames, self.c1, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1 = activation
        self.conv2 = nn.Conv2d(self.c1, self.c2, kernel_size=3)
        self.act2 = activation

        self.film_spatial_enabled = False
        self.film_spatial_sites = ["conv1", "conv2"]
        self.film1 = FiLM2d(self.prop_dim, self.c1, activation=activation)
        self.film2 = FiLM2d(self.prop_dim, self.c2, activation=activation)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.c2 * 25 * 39, 128)
        self.act3 = activation
        self.fc2 = nn.Linear(128, scandots_output_dim)
        self.output_activation = nn.Tanh() if output_activation == "tanh" else activation

    def _forward_features(self, x: torch.Tensor, proprio: torch.Tensor = None) -> torch.Tensor:
        x = self.conv1(x)
        if self.film_spatial_enabled and proprio is not None and "conv1" in self.film_spatial_sites:
            x = self.film1(x, proprio)
        x = self.pool(x)
        x = self.act1(x)
        x = self.conv2(x)
        if self.film_spatial_enabled and proprio is not None and "conv2" in self.film_spatial_sites:
            x = self.film2(x, proprio)
        x = self.act2(x)
        return x

    def forward(self, images: torch.Tensor, proprio: torch.Tensor = None):
        if images.dim() == 3:
            x = images.unsqueeze(1)
        elif images.dim() == 4:
            x = images
        else:
            raise ValueError(f"Unsupported depth image shape: {images.shape}")
        x = self._forward_features(x, proprio)
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        return self.output_activation(self.fc2(x))


# ---------------------------------------------------------------------------
# MambaTemporalEncoder  —  Mamba backbone for temporal modelling (Eq. 3)
# ---------------------------------------------------------------------------

class MambaTemporalEncoder(nn.Module):
    """Encode depth history (B, F, 32) → latent (B, 32) via Mamba SSM layers.

    Optionally returns a Mamba state projection for auxiliary distillation.
    """

    def __init__(self, history_len: int, proprio_dim: int,
                 d_state: int = 64, n_layers: int = 1,
                 dropout: float = 0.0, activation: nn.Module = nn.ELU(),
                 pos_enc: bool = False):
        super().__init__()
        self.history_len = int(max(1, history_len))
        self.pos_enc = None
        if pos_enc and history_len > 1:
            self.pos_enc = nn.Parameter(torch.zeros(int(history_len), 32))

        self.layers = nn.ModuleList([
            SimpleMambaBlock(d_model=32, d_state=int(d_state),
                             dropout=float(dropout), proprio_dim=int(proprio_dim))
            for _ in range(max(1, int(n_layers)))
        ])
        self.out_proj = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.state_proj = nn.Sequential(nn.Linear(32, 64), activation, nn.Linear(64, 32))

    def forward(self, per_frame: torch.Tensor, proprio: torch.Tensor):
        if per_frame.dim() != 3 or per_frame.size(-1) != 32:
            raise ValueError(f"per_frame expected (B,F,32), got {tuple(per_frame.shape)}")
        B, F, _ = per_frame.shape
        if F < self.history_len:
            per_frame = torch.cat([per_frame, per_frame[:, -1:].repeat(1, self.history_len - F, 1)], dim=1)
        elif F > self.history_len:
            per_frame = per_frame[:, -self.history_len:]

        x = per_frame
        if self.pos_enc is not None:
            x = x + self.pos_enc[:x.size(1)]
        for layer in self.layers:
            x = layer(x, proprio)

        last = x[:, -1, :]
        return self.out_proj(last), self.state_proj(last)


# ---------------------------------------------------------------------------
# RecurrentDepthBackbone  —  FiLM-CNN → Mamba → GRU
# ---------------------------------------------------------------------------

class RecurrentDepthBackbone(nn.Module):
    """Student depth encoder pipeline: FiLM-CNN → Mamba → combination MLP → GRU.

    When ``temporal_cfg`` is None the backbone falls back to a simple
    combination-MLP + GRU pipeline (no Mamba / FiLM), used for baselines.
    """

    def __init__(self, base_backbone, env_cfg, temporal_cfg=None):
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.classic_mode = temporal_cfg is None
        self.temporal_cfg = None if self.classic_mode else dict(temporal_cfg)
        self.latest_mamba_state = None

        proprio_dim = 53 if env_cfg is None else int(env_cfg.env.n_proprio)

        if not self.classic_mode:
            self.film_spatial_enabled = bool(self.temporal_cfg.get("film_spatial_enabled", False))
            self.base_backbone.film_spatial_enabled = self.film_spatial_enabled
            self.base_backbone.film_spatial_sites = list(
                self.temporal_cfg.get("film_spatial_sites", ["conv1", "conv2"])
            )

            default_hist = 1
            if env_cfg is not None:
                env_obj = getattr(env_cfg, "env", None)
                if env_obj is not None:
                    default_hist = int(
                        getattr(env_obj, "depth_history_len", None)
                        or getattr(env_obj, "scan_history_len", 1)
                    )
            self.depth_history_len = int(self.temporal_cfg.get("depth_history_len", default_hist))
            self.depth_use_combination_mlp = bool(self.temporal_cfg.get("depth_use_combination_mlp", True))

            self.temporal_encoder = MambaTemporalEncoder(
                history_len=self.depth_history_len,
                proprio_dim=proprio_dim,
                d_state=int(self.temporal_cfg.get("depth_mamba_d_state", 64)),
                n_layers=int(self.temporal_cfg.get("depth_mamba_layers", 1)),
                dropout=float(self.temporal_cfg.get("depth_mamba_dropout", 0.0)),
                activation=activation,
            )
        else:
            self.depth_use_combination_mlp = True
            self.temporal_encoder = None

        n_prop = proprio_dim
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_prop, 128), activation, nn.Linear(128, 32),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(nn.Linear(512, 34), nn.Tanh())
        self.hidden_states = None

    def forward(self, depth_image, proprioception, applied_action=None):
        self.latest_mamba_state = None
        use_film = not self.classic_mode and self.film_spatial_enabled
        proprio_for_cnn = proprioception if use_film else None

        if self.classic_mode:
            depth_latent = self.base_backbone(depth_image)
        else:
            if depth_image.dim() == 3:
                B, H, W = depth_image.shape
                F = 1
                flat = depth_image
            elif depth_image.dim() == 4:
                B, F, H, W = depth_image.shape
                flat = depth_image.reshape(B * F, H, W)
            else:
                raise ValueError(f"Unexpected depth_image shape: {tuple(depth_image.shape)}")

            per_frame = self.base_backbone(flat, proprio_for_cnn).reshape(B, F, 32)
            depth_latent, m_state = self.temporal_encoder(per_frame, proprioception)
            self.latest_mamba_state = m_state

        if self.depth_use_combination_mlp:
            depth_latent = self.combination_mlp(torch.cat((depth_latent, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        return self.output_mlp(depth_latent.squeeze(1))

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
