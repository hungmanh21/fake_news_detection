import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, yaml, torch, os
from src.data.dataset import NewsDataLoader
from src.models.baseline import SimpleBertClassifier
from src.train.simple_trainer import Trainer

from src.utils import get_logger

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger = get_logger()
    logger.info("Loaded configuration:")
    for k, v in cfg.items():
        logger.info(f"{k}: {v}")

    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # init wandb if available
    wandb_tracking = True
    if cfg.get('wandb', {}).get('enabled', False):
        from src.tracking.wandb import init_wandb
        run = init_wandb(cfg['wandb']['run_name'], cfg['wandb']['project'], cfg)
        if run is None:
            logger.warning("WandB initialization failed. Continuing without logging.")
            wandb_tracking = False
    else:
        logger.info("WandB logging is disabled.")
        wandb_tracking = False

    # Load data
    data_loader = NewsDataLoader(cfg['data']['csv_file'],
                                 test_size=cfg['data']['test_size'],
                                 batch_size=cfg['data']['batch_size'],
                                 tokenizer_name=cfg['tokenizer']['name'],
                                 max_length=cfg['tokenizer']['max_length'],
                                 seed=cfg['seed'])
    train_loader, val_loader = data_loader.get_loaders()
    model = SimpleBertClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    trainer = Trainer(model, optimizer, criterion, device, wandb_tracking)

    best_val_acc = 0.0
    os.makedirs(cfg['train']['save_dir'], exist_ok=True)

    for epoch in range(cfg['num_epochs']):
        train_loss, train_acc, train_precision, train_recall, train_f1 = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_precision, val_recall, val_f1 = trainer.eval_epoch(val_loader)

        logger.info(
            f"Epoch {epoch + 1}/{cfg['num_epochs']} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
            f"precision: {train_precision:.4f}, recall: {train_recall:.4f}, f1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, "
            f"precision: {val_precision:.4f}, recall: {val_recall:.4f}, f1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(cfg['train']['save_dir'], 'best_model.pt'))
            logger.info("✅ Saved new best model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)