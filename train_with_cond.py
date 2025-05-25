from collections import OrderedDict
from tqdm import tqdm
import argparse

from dataset.cad_dataset import get_dataloader_with_cond
from config import ConfigAE
from utils import cycle
from trainer import TrainerAEWithCond

def main():
    # Load config for conditional AE
    cfg = ConfigAE('train')

    # Initialize trainer
    tr_agent = TrainerAEWithCond(cfg)

    # Load from checkpoint if continuing
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # Create dataloaders for training and validation
    train_loader = get_dataloader_with_cond('train', cfg)
    val_loader = get_dataloader_with_cond('validation', cfg)
    val_loader_all = get_dataloader_with_cond('validation', cfg)
    val_loader_cycle = cycle(val_loader)

    clock = tr_agent.clock
    # tr_agent.evaluate(val_loader_all)
    for e in range(clock.epoch, cfg.nr_epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.train_func(data)

            if losses is None:
                continue  # Skip if batch was invalid (e.g., NaN in physical predictions)

            pbar.set_description(f"EPOCH[{e}][{b}]")
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # Run validation step occasionally
            if clock.step % cfg.val_frequency == 0:
                val_data = next(val_loader_cycle)
                val_outputs, val_losses = tr_agent.val_func(val_data)
                if val_losses is None:
                    continue  # Skip validation batch if invalid

            clock.tick()
            tr_agent.update_learning_rate()

        # Full evaluation every N epochs
        if clock.epoch % 5 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        # Save checkpoints
        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
