def accuracy(preds, targets):
    correct = (preds == targets).sum().item()
    return correct / len(targets)
