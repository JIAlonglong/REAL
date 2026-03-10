# flake8: noqa
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims)
        self.predict_uncertainty = bool(kwargs.get("predict_uncertainty", False))
        activation = get_activation(activation)

        # Shared trunk
        layers = []
        last = int(self.input_dim)
        for h in list(hidden_dims):
            layers.append(nn.Linear(last, int(h)))
            layers.append(activation)
            last = int(h)
        self.trunk = nn.Sequential(*layers)
        # Mean head (always)
        self.mean_head = nn.Linear(last, int(self.output_dim))
        # Optional log-variance head (diagonal Gaussian)
        self.logvar_head = nn.Linear(last, int(self.output_dim)) if self.predict_uncertainty else None

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compatible loader.

        Old checkpoints (before 2026-01) saved Estimator as a single Sequential under key prefix:
            estimator.0.weight/bias, estimator.2.weight/bias, ..., estimator.{2*m}.weight/bias
        where m = len(hidden_dims).

        New Estimator uses:
            trunk.{0,2,4,...}.weight/bias and mean_head.weight/bias (plus optional logvar_head.*)

        This function auto-maps old keys to the new structure so old runs can resume.
        """
        try:
            keys = list(state_dict.keys())
        except Exception:
            keys = []

        has_old = any(k.startswith("estimator.") for k in keys)
        has_new = any(k.startswith("trunk.") or k.startswith("mean_head.") for k in keys)

        # Map legacy Sequential weights -> (trunk linears + mean_head)
        if has_old and (not has_new):
            mapped = {}
            m = len(self.hidden_dims)
            for k, v in state_dict.items():
                if not k.startswith("estimator."):
                    # ignore unrelated keys if any
                    continue
                parts = k.split(".")
                if len(parts) < 3:
                    continue
                try:
                    layer_idx = int(parts[1])
                except Exception:
                    continue
                suffix = ".".join(parts[2:])  # weight / bias

                # Only Linear layers are stored at even indices in the legacy Sequential.
                if layer_idx % 2 != 0:
                    continue
                lin_k = layer_idx // 2

                if lin_k < m:
                    # trunk: [Linear, act, Linear, act, ...] => linear at 2*lin_k
                    mapped[f"trunk.{2 * lin_k}.{suffix}"] = v
                elif lin_k == m:
                    mapped[f"mean_head.{suffix}"] = v

            # If uncertainty is enabled, older checkpoints won't have logvar_head.*
            # So we must allow missing keys.
            strict2 = False if self.predict_uncertainty else strict
            return super().load_state_dict(mapped, strict=strict2)

        # New-format checkpoints: use default behavior (but allow missing logvar_head when enabled).
        if not self.predict_uncertainty:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logvar_head.")}
        strict2 = False if self.predict_uncertainty else strict
        return super().load_state_dict(state_dict, strict=strict2)
    
    def forward(self, input):
        # By default return mean only (keeps backward compatibility with existing RL pipeline)
        h = self.trunk(input)
        return self.mean_head(h)

    def forward_with_uncertainty(self, input):
        """Return (mean, log_var) for diagonal Gaussian modeling."""
        h = self.trunk(input)
        mean = self.mean_head(h)
        if self.logvar_head is None:
            raise RuntimeError("Estimator was created with predict_uncertainty=False")
        log_var = self.logvar_head(h)
        return mean, log_var
    
    def inference(self, input):
        with torch.no_grad():
            return self.forward(input)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, activation=nn.ReLU):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = activation()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = activation()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_hidden_channels, kernel_size=2, dropout=0.2, activation=nn.ReLU):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    activation=activation,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TcnEstimator(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        activation="ReLU",
        **kwargs,
    ):
        super(TcnEstimator, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.predict_uncertainty = bool(kwargs.get("predict_uncertainty", False))
        act = getattr(nn, activation, nn.ReLU)
        self.tcn = TemporalConvNet(
            self.input_dim,
            list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
            activation=act,
        )
        self.mean_head = nn.Linear(int(num_channels[-1]), self.output_dim)
        self.logvar_head = nn.Linear(int(num_channels[-1]), self.output_dim) if self.predict_uncertainty else None
        self.output_log_std = bool(self.predict_uncertainty)
        self.sequence_input = True
        self.init_weights()

    def init_weights(self):
        self.mean_head.weight.data.normal_(0, 0.01)
        if self.logvar_head is not None:
            self.logvar_head.weight.data.normal_(0, 0.01)

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self.predict_uncertainty:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logvar_head.")}
        strict2 = False if self.predict_uncertainty else strict
        return super().load_state_dict(state_dict, strict=strict2)

    def _prep(self, x):
        if x.dim() == 3:
            if x.shape[1] == self.input_dim:
                return x
            return x.permute(0, 2, 1)
        if x.dim() == 2:
            return x.unsqueeze(2)
        raise RuntimeError("Invalid input shape for TcnEstimator")

    def forward(self, input):
        x = self._prep(input)
        x = self.tcn(x)
        feat = x[:, :, -1]
        return self.mean_head(feat)

    def forward_with_uncertainty(self, input):
        x = self._prep(input)
        x = self.tcn(x)
        feat = x[:, :, -1]
        mean = self.mean_head(feat)
        if self.logvar_head is None:
            raise RuntimeError("Estimator was created with predict_uncertainty=False")
        log_var = self.logvar_head(feat)
        log_std = 0.5 * log_var
        return mean, log_std

    def inference(self, input):
        with torch.no_grad():
            return self.forward(input)

