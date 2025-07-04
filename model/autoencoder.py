from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask


class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        self.command_embed = nn.Embedding(cfg.n_commands, cfg.d_model)

        args_dim = cfg.args_dim + 1
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        S, N = commands.shape

        src = self.command_embed(commands.long()) + \
              self.embed_fcn(self.arg_embed((args + 1).long()).view(S, N, -1))  # shift due to -1 PAD_VAL

        if self.use_group:
            src = src + self.group_embed(groups.long())

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None

        src = self.embedding(commands, args, group_mask)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)

        z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z)
        return z


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z):
        src = self.embedding(z)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits


class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)

class BottleneckWithCond(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_z),
            nn.Tanh()
        )
        self.cond_fc = nn.Linear(cfg.cond_dim, cfg.cond_embed_dim)  # New
        self.project_z = nn.Linear(cfg.dim_z + cfg.cond_embed_dim, cfg.dim_z)  # Combine z + cond

    def forward(self, z, cond):
        z = self.bottleneck(z)  # (1, B, dim_z)
        cond_embed = self.cond_fc(cond)  # (B, cond_embed_dim)
        cond_embed = cond_embed.unsqueeze(0)  # (1, B, cond_embed_dim)
        z_cond = torch.cat([z, cond_embed], dim=-1)  # (1, B, dim_z + cond_embed_dim)
        return self.project_z(z_cond)  # (1, B, dim_z)

class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)

    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None

        if z is None:
            z = self.encoder(commands_enc_, args_enc_)
            z = self.bottleneck(z)
        else:
            z = _make_seq_first(z)

        if encode_mode: return _make_batch_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],
            "args_logits": out_logits[1]
        }

        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res

class ConditionalVAEBottleneck(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc_mean = nn.Linear(cfg.d_model, cfg.dim_z)
        self.fc_logvar = nn.Linear(cfg.d_model, cfg.dim_z)
        self.cond_fc = nn.Linear(cfg.cond_dim, cfg.cond_embed_dim)
        self.project_z = nn.Linear(cfg.dim_z + cfg.cond_embed_dim, cfg.dim_z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, z, cond):
        mu = self.fc_mean(z)
        logvar = self.fc_logvar(z)
        z_sampled = self.reparameterize(mu, logvar)

        cond_embed = self.cond_fc(cond).unsqueeze(0)  # (1, B, cond_embed_dim)
        z_cond = torch.cat([z_sampled, cond_embed], dim=-1)  # (1, B, dim_z + cond_embed_dim)
        return self.project_z(z_cond), mu, logvar


class DecoderWithCond(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        self.cond_fc = nn.Linear(cfg.cond_dim, cfg.cond_embed_dim)  # Map condition vector to embedding

        # Transformer decoder layer with two memories (latent + cond)
        self.decoder_layer = TransformerDecoderLayerGlobalImproved(
            d_model=cfg.d_model,
            d_global=cfg.dim_z,
            d_global2=cfg.cond_embed_dim,  # Enables memory2 path
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout
        )

        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(
            self.decoder_layer,
            num_layers=cfg.n_layers_decode,
            norm=self.decoder_norm
        )

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z, cond):
        """
        Args:
            z: latent (1, B, dim_z)
            cond: condition vector (B, cond_dim)
        """
        # Project cond to cond_embed and reshape to match (1, B, cond_embed_dim)
        cond_embed = self.cond_fc(cond).unsqueeze(0)

        # Positional input for decoder
        src = self.embedding(z)  # (S, B, d_model)

        # Decode using both z and cond_embed
        out = self.decoder(
            tgt=src,               # (S, B, d_model)
            memory=z,              # latent vector
            memory2=cond_embed     # conditioning vector
        )

        # Generate logits
        command_logits, args_logits = self.fcn(out)
        return command_logits, args_logits

class CADTransformerWithCond(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.bottleneck = ConditionalVAEBottleneck(cfg)
        self.decoder = DecoderWithCond(cfg)

    def forward(self, commands_enc, args_enc, cond, z=None, return_tgt=True, encode_mode=False):
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)

        if z is None:
            z_encoded = self.encoder(commands_enc_, args_enc_)
            z, mu, logvar = self.bottleneck(z_encoded, cond)
        else:
            z = _make_seq_first(z)
            mu, logvar = None, None

        if encode_mode:
            return _make_batch_first(z)

        command_logits, args_logits = self.decoder(z, cond)
        command_logits, args_logits = _make_batch_first(command_logits, args_logits)

        res = {
            "command_logits": command_logits,
            "args_logits": args_logits,
            "mu": mu,
            "logvar": logvar,
            "cond": cond
        }


        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res