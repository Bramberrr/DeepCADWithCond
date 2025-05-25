from collections import OrderedDict
from tqdm import tqdm
import argparse

from dataset.cad_dataset import get_dataloader_with_cond
from config import ConfigAE
from utils import cycle
from trainer import TrainerRegressor


def main():
    # Load config
    cfg = ConfigAE('train')

    # Initialize trainer for regressor
    tr_agent = TrainerRegressor(cfg)

    # Load from checkpoint if resuming
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # Load dataloaders
    train_loader = get_dataloader_with_cond('train', cfg)
    val_loader = get_dataloader_with_cond('validation', cfg)
    val_loader_cycle = cycle(val_loader)

    # Begin training
    clock = tr_agent.clock
    for e in range(clock.epoch, cfg.nr_epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.train_func(data)

            pbar.set_description(f"[REG][EPOCH {e}][{b}]")
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # Periodic validation
            if clock.step % cfg.val_frequency == 0:
                val_data = next(val_loader_cycle)
                _, val_losses = tr_agent.val_func(val_data)

            clock.tick()
            tr_agent.update_learning_rate()

        # Epoch end
        clock.tock()
        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt("latest")


if __name__ == '__main__':
    main()
