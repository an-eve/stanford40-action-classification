import time
import torch
from evaluation import compute_accuracy_loss, compute_accuracy


def train_model(model, num_epochs, train_loader, valid_loader,
                criterion, optimizer, device, scheduler=None):

    start_time = time.time()
    train_loss_list, valid_loss_list, train_acc_list, valid_acc_list = [], [], [], []

    for epoch in range(num_epochs):

        minibatch_loss_list= []
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_loss = sum(minibatch_loss_list)/len(minibatch_loss_list)
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc, valid_loss = compute_accuracy_loss(model, valid_loader, criterion, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train Loss: {train_loss :.4f} '
                  f'| Validation Loss: {valid_loss :.4f}\n'
                  f'| Train Accuracy: {train_acc :.2f}% '
                  f'| Validation Accuracy: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None:
            scheduler.step(valid_acc_list[-1])


    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list
