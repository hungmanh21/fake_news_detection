import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

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
        all_targets, all_predictions = [], []

        for batch in tqdm(loader, desc='Train', leave=False):
            inputs = {
                "input_ids": batch['input_ids'].to(self.device),
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

            # Collect predictions and targets for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        # Log to wandb if available
        if self.wandb:
            wandb.log({
                'train_loss': running_loss / total,
                'train_accuracy': correct / total,
                'train_precision': precision,
                'train_recall': recall,
                'train_f1': f1
            })

        return running_loss / total, correct / total, precision, recall, f1

    def eval_epoch(self, loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_targets, all_predictions = [], []

        with torch.no_grad():
            for batch in loader:
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                }
                targets = batch['labels'].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Collect predictions and targets for metrics calculation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate metrics
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        # Log to wandb if available
        if self.wandb:
            wandb.log({
                'eval_loss': running_loss / total,
                'eval_accuracy': correct / total,
                'eval_precision': precision,
                'eval_recall': recall,
                'eval_f1': f1
            })

        return running_loss / total, correct / total, precision, recall, f1
