import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBiGRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
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

    def forward(self, x):

        y, _ = self.gru(x)
        y = self.dropout(y)

        if self.proj is not None:
            x = self.proj(x)

        out = x + y
        out = self.norm(out)
        return out


class CompactBiGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        n_pitches=88,
        proj_dim=96,
        hidden_dim=160,
        num_blocks=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, proj_dim)

        blocks = []
        dim = proj_dim
        for _ in range(num_blocks):
            block = ResidualBiGRUBlock(dim, hidden_dim, dropout=dropout)
            blocks.append(block)
            dim = block.out_dim

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(dim, n_pitches)

    def forward(self, x):
        
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2)
        elif x.dim() == 3:

            x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = torch.relu(x)

        x = self.blocks(x)

        logits = self.head(x)
        return logits