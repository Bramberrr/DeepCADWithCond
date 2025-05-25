import torch
import torch.nn as nn
from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .autoencoder import CADEmbedding

class PhysicalRegressorFromLogits(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_commands = cfg.n_commands
        self.n_args = cfg.n_args
        self.args_dim = cfg.args_dim + 1

        # These handle soft logits instead of discrete indices
        self.command_fc = nn.Linear(self.n_commands, self.d_model)
        self.arg_fc = nn.Linear(self.n_args * self.args_dim, self.d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=cfg.n_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # predict [volume, thickness]
        )

    def forward(self, cmd_logits, args_logits):
        # Inputs: (B, S, C), (B, S, A, D)
        B, S, C = cmd_logits.shape
        _, _, A, D = args_logits.shape

        cmd_emb = self.command_fc(cmd_logits)            # (B, S, d_model)
        args_flat = args_logits.view(B, S, A * D)         # (B, S, A*D)
        args_emb = self.arg_fc(args_flat)                 # (B, S, d_model)

        x = cmd_emb + args_emb                           # (B, S, d_model)
        x = self.encoder(x)                              # (B, S, d_model)
        pooled = x.mean(dim=1)                           # (B, d_model)
        return self.fc(pooled)                           # (B, 2)
