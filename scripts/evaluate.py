import argparse, yaml, torch, os
from src.data.dataset import get_loaders
from src.models.simple_cnn import SimpleCNN
from src.train.simple_trainer import Trainer

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = get_loaders(cfg['data_dir'], cfg['batch_size'])
    model = SimpleCNN().to(device)

    checkpoint_path = os.path.join(cfg['train']['save_dir'], 'best_model.pt')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    trainer = Trainer(model, None, torch.nn.CrossEntropyLoss(), device)
    test_loss, test_acc = trainer.eval_epoch(test_loader)

    print(f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
