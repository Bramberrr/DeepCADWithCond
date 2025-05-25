import torch
import torch.nn as nn
import torch.nn.functional as F
from cadlib.macro import CMD_ARGS_MASK
from model.model_utils import _get_padding_mask, _get_visibility_mask

class CADLoss(nn.Module):
    def __init__(self, cfg, physical_regressor):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.n_args = cfg.n_args
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

        self.physical_regressor = physical_regressor
        for p in self.physical_regressor.parameters():
            p.requires_grad = False

        self.material_densities = torch.tensor([2700, 7850, 8900, 7190, 8960, 19300], dtype=torch.float32).cuda()

    def forward(self, output):
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]
        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]
        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long()
        )

        loss_args = F.cross_entropy(
            args_logits[mask.bool()].reshape(-1, self.args_dim),
            tgt_args[mask.bool()].reshape(-1).long() + 1
        )

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        loss_total = loss_cmd + loss_args
        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}

        # KL loss for VAE
        if "mu" in output and output["mu"] is not None and "logvar" in output and output["logvar"] is not None:
            mu, logvar = output["mu"], output["logvar"]
            loss_kl = self.weights["loss_kl_weight"] * self.kl_divergence(mu, logvar)
            loss_total += loss_kl
            res["loss_kl"] = loss_kl

        # Physical constraint loss via logits-based regressor
        if "cond" in output:
            cond = output["cond"]
            B = cond.shape[0]

            cmd_prob = F.softmax(command_logits, dim=-1)  # (B, S, C)
            args_prob = F.softmax(args_logits, dim=-1)     # (B, S, A, D)

            pred_phys = self.physical_regressor(cmd_prob, args_prob)  # (B, 2)
            pred_volume, pred_thickness = pred_phys[:, 0], pred_phys[:, 1]

            # Material is one-hot from cond[3:], use it to get density
            material_logits = cond[:, 3:]  # (B, 6)
            density = (material_logits * self.material_densities).sum(dim=-1)  # (B,)
            inferred_volume_from_mass = cond[:, 1] / (density + 1e-6)
            min_volume = torch.min(cond[:, 0], inferred_volume_from_mass)
            min_thickness = cond[:, 2]

            # Physical losses (hinge)
            loss_vol = F.relu(pred_volume - min_volume).mean()
            loss_thick = F.relu(min_thickness - pred_thickness).mean()
            loss_phys = self.weights.get("loss_phys_weight", 1.0) * (loss_vol + loss_thick)
            loss_total += loss_phys
            res["loss_phys"] = loss_phys

        res["loss_total"] = loss_total
        return res

    @staticmethod
    def kl_divergence(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from model.model_utils import _get_padding_mask, _get_visibility_mask
# from cadlib.macro import CMD_ARGS_MASK

# class CADLoss(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.n_commands = cfg.n_commands
#         self.args_dim = cfg.args_dim + 1
#         self.weights = cfg.loss_weights

#         self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

#     def forward(self, output):
#         tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]
#         visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
#         padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

#         command_logits, args_logits = output["command_logits"], output["args_logits"]
#         mask = self.cmd_args_mask[tgt_commands.long()]

#         loss_cmd = F.cross_entropy(
#             command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
#             tgt_commands[padding_mask.bool()].reshape(-1).long()
#         )

#         loss_args = F.cross_entropy(
#             args_logits[mask.bool()].reshape(-1, self.args_dim),
#             tgt_args[mask.bool()].reshape(-1).long() + 1  # shift due to -1 PAD_VAL
#         )

#         loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
#         loss_args = self.weights["loss_args_weight"] * loss_args

#         loss_total = loss_cmd + loss_args
#         res = {"loss_cmd": loss_cmd, "loss_args": loss_args}

#         # Optional KL loss for VAE
#         if "mu" in output and output["mu"] is not None and "logvar" in output and output["logvar"] is not None:
#             mu, logvar = output["mu"], output["logvar"]
#             loss_kl = self.weights["loss_kl_weight"] * self.kl_divergence(mu, logvar)
#             loss_total += loss_kl
#             res["loss_kl"] = loss_kl

#         res["loss_total"] = loss_total
#         return res

#     @staticmethod
#     def kl_divergence(mu, logvar):
#         return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(1)
