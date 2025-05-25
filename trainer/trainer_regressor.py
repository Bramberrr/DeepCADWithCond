import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base import BaseTrainer
from model.regressor import PhysicalRegressorFromLogits
from .scheduler import GradualWarmupScheduler
from cadlib.macro import CMD_ARGS_MASK


class TrainerRegressor(BaseTrainer):
    def build_net(self, cfg):
        self.net = PhysicalRegressorFromLogits(cfg).cuda()

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = nn.MSELoss().cuda()

    def get_teacher_logits(self, command, args, cfg):
        """
        One-hot encode ground truth commands and args to simulate logits (for training).
        """
        B, S = command.shape
        device = command.device

        cmd_logits = torch.zeros(B, S, cfg.n_commands, device=device)
        cmd_logits.scatter_(2, command.unsqueeze(-1), 1.0)

        args_logits = torch.zeros(B, S, cfg.n_args, cfg.args_dim + 1, device=device)
        for i in range(cfg.n_args):
            valid_mask = (args[:, :, i] != -1).unsqueeze(-1)
            idx = (args[:, :, i] + 1).clamp(min=0).unsqueeze(-1)
            args_logits[:, :, i].scatter_(2, idx, valid_mask.float())

        return cmd_logits, args_logits

    def forward(self, data):
        command = data['command'].cuda()  # (B, S)
        args = data['args'].cuda()        # (B, S, A)
        cond = data['cond'].cuda()        # (B, 9)

        cmd_logits, args_logits = self.get_teacher_logits(command, args, self.cfg)
        preds = self.net(cmd_logits, args_logits)  # (B, 2)

        targets = cond[:, [0, 2]]  # volume, thickness
        loss = self.loss_func(preds, targets)
        return preds, {"reg_loss": loss}

    def evaluate(self, test_loader):
        self.net.eval()
        losses = []

        for data in tqdm(test_loader, desc="Evaluating RegressorFromLogits"):
            with torch.no_grad():
                _, loss_dict = self.forward(data)
                losses.append(loss_dict["reg_loss"].item())

        avg_loss = sum(losses) / len(losses)
        print(f"[Eval LogitRegressor] Average MSE: {avg_loss:.6f}")
        return avg_loss
