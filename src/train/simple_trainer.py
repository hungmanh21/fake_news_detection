import torch
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, optimizer, criterion, device, wandb: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.wandb = wandb

    def train_epoch(self, loader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(loader, desc='Train', leave=False):
            inputs = {
                "input_ids" : batch['input_ids'].to(self.device),
                "attention_mask": batch['attention_mask'].to(self.device),
            }
            targets = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # log to wandb if available
        if self.wandb:
            wandb.log({
                'train_loss': running_loss / total,
                'train_accuracy': correct / total
            })

        return running_loss / total, correct / total

    def eval_epoch(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return running_loss / total, correct / total
