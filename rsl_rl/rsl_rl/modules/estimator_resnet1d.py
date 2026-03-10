import torch
import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetEstimator1D(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        layers=(2, 2, 2, 2),
        base_planes=64,
        predict_uncertainty=False,
        input_kernel_size=7,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.predict_uncertainty = predict_uncertainty
        self.sequence_input = True
        k = int(input_kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        self.inplanes = base_planes
        self.input_block = nn.Sequential(
            nn.Conv1d(self.input_dim, base_planes, kernel_size=k, stride=2, padding=k // 2, bias=False),
            nn.BatchNorm1d(base_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, self.output_dim)
        if self.predict_uncertainty:
            self.fc_logstd = nn.Linear(512 * BasicBlock1D.expansion, self.output_dim)
        self._initialize()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock1D):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # x shape: (N, L, C) -> permute to (N, C, L)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            # If input is (N, C), treat as (N, C, 1)
            x = x.unsqueeze(2)
        
        x = self.input_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.predict_uncertainty:
            return self.fc(x), self.fc_logstd(x)
        return self.fc(x)

    def forward_with_uncertainty(self, x):
        """Return (mean, log_std) for uncertainty-aware estimation.

        When predict_uncertainty is True, forward() already returns a tuple.
        Otherwise, returns zeros as a fallback for log_std.
        """
        out = self.forward(x)
        if isinstance(out, tuple):
            return out
        return out, torch.zeros_like(out)
