import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResBlock2D(nn.Module):

    def __init__(self, in_ch, out_ch, stride=(1, 1)):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.skip = None
        if stride != (1, 1) or in_ch != out_ch:
            self.skip = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        residual = self.skip(x) if self.skip is not None else x
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class PreActResBlock1D(nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        residual = x
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class CNNTemporal(nn.Module):

    def __init__(
        self,
        n_freq_bins=229,
        n_pitches=88,
        cnn_channels=(32, 64, 96, 128),
        temporal_blocks=3
    ):
        super().__init__()

        self.n_pitches = n_pitches
        self.n_freq_bins = n_freq_bins

        self.stem = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels[0],
            kernel_size=3,
            padding=1,
            bias=False
        )


        blocks = []
        in_ch = cnn_channels[0]

        for i, ch in enumerate(cnn_channels):
            if i < len(cnn_channels) - 1:
                stride = (2, 1)
            else:
                stride = (1, 1)

            blocks.append(
                PreActResBlock2D(
                    in_ch,
                    ch,
                    stride=stride
                )
            )
            in_ch = ch

        self.cnn2d = nn.Sequential(*blocks)
        self.bn_final_2d = nn.BatchNorm2d(in_ch)

        temporal_ch = in_ch

        tblocks = []
        for i in range(temporal_blocks):
            dilation = 2 ** i
            tblocks.append(
                PreActResBlock1D(
                    channels=temporal_ch,
                    kernel_size=3,
                    dilation=dilation
                )
            )

        self.temporal = nn.Sequential(*tblocks)

        self.head = nn.Conv1d(
            in_channels=temporal_ch,
            out_channels=n_pitches,
            kernel_size=1,
            bias=True
        )

    def forward(self, spec):

        x = self.stem(spec)
        x = self.cnn2d(x)
        x = self.bn_final_2d(x)
        x = F.relu(x)

        x = x.mean(dim=2)

        x = self.temporal(x)

        logits = self.head(x)
        logits = logits.permute(0, 2, 1)
        
        return logits                         