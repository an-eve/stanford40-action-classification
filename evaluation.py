import torch


def compute_accuracy(model, data_loader, device="cuda"):
    correct_pred, num_examples = 0, 0
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_accuracy_loss(model, data_loader, criterion, device="cuda"):
    correct_pred, num_examples, loss = 0, 0, 0
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            targets_float = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets_float).sum()

            loss_batch = criterion(logits, targets)
            loss += loss_batch.item()

    return correct_pred.float()/num_examples * 100, loss/len(data_loader)
