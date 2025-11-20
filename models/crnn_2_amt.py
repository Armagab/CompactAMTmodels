import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResBlock2D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.skip = None
        if stride != (1, 1) or in_ch != out_ch:
            self.skip = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.bn1(x)
        out = F.relu(out)
        residual = self.skip(x) if self.skip is not None else x

        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return out + residual


class ResidualBiGRUBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 2 * hidden_dim

        self.proj = None
        if input_dim != self.out_dim:
            self.proj = nn.Linear(input_dim, self.out_dim)

        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y, _ = self.gru(x)
        y = self.dropout(y)

        if self.proj is not None:
            x = self.proj(x)

        out = x + y
        out = self.norm(out)
        return out


class CRNNCompactFO(nn.Module):
    
    def __init__(
        self,
        n_freq_bins=229,
        n_pitches=88,
        cnn_channels=(32, 64, 96, 128),
        onset_hidden=128,
        frame_hidden=160,
        onset_blocks=1,
        frame_blocks=2,
        rnn_dropout=0.2,
    ) -> None:
        super().__init__()

        self.n_freq_bins = n_freq_bins
        self.n_pitches = n_pitches

        # CNN ENCODER
        self.stem = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels[0],
            kernel_size=3,
            padding=1,
            bias=False,
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
                    in_ch=in_ch,
                    out_ch=ch,
                    stride=stride,
                )
            )
            in_ch = ch

        self.cnn2d = nn.Sequential(*blocks)
        self.bn_final_2d = nn.BatchNorm2d(in_ch)
        self.cnn_out_ch = in_ch

        # Onset-head 
        onset_rnn_blocks = []
        onset_dim = self.cnn_out_ch
        for _ in range(onset_blocks):
            block = ResidualBiGRUBlock(
                input_dim=onset_dim,
                hidden_dim=onset_hidden,
                dropout=rnn_dropout,
            )
            onset_rnn_blocks.append(block)
            onset_dim = block.out_dim

        self.onset_rnn = nn.Sequential(*onset_rnn_blocks)
        self.onset_head = nn.Linear(onset_dim, n_pitches)

        # Frame-head
        frame_rnn_blocks = []
        frame_dim = self.cnn_out_ch + n_pitches
        for _ in range(frame_blocks):
            block = ResidualBiGRUBlock(
                input_dim=frame_dim,
                hidden_dim=frame_hidden,
                dropout=rnn_dropout,
            )
            frame_rnn_blocks.append(block)
            frame_dim = block.out_dim

        self.frame_rnn = nn.Sequential(*frame_rnn_blocks)
        self.frame_head = nn.Linear(frame_dim, n_pitches)

    def forward(self, spec: torch.Tensor):

        B, _, n_freq, T = spec.shape

        x = self.stem(spec)

        x = self.cnn2d(x)
        x = self.bn_final_2d(x)
        x = F.relu(x)

        x = x.mean(dim=2)

        feat_seq = x.transpose(1, 2).contiguous()

        # Onset-head 
        onset_seq = feat_seq
        if len(self.onset_rnn) > 0:
            onset_seq = self.onset_rnn(onset_seq)
        onset_logits = self.onset_head(onset_seq)
        onset_probs = torch.sigmoid(onset_logits)

        # Frame-head
        frame_in = torch.cat([feat_seq, onset_probs], dim=-1)

        frame_seq = frame_in
        if len(self.frame_rnn) > 0:
            frame_seq = self.frame_rnn(frame_seq)

        frame_logits = self.frame_head(frame_seq)

        return onset_logits, frame_logits