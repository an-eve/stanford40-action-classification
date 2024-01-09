import matplotlib.pyplot as plt
import numpy as np
import torch


def undo_normalize(tensor, mean=(0.485, 0.456, 0.406), std= (0.229, 0.224, 0.225)):

    mean = torch.tensor(mean)
    std = torch.tensor(std)
    original_tensor = tensor.permute(1,2,0) * std + mean
    return original_tensor
    

def plot_random_images(dataset, classes, model=None, device="cuda", num_images=5): #classes - list of labels' names

    if model:
        model.eval()

        test_indices = np.random.randint(0, len(dataset), size=num_images)

        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

        for i, idx in enumerate(test_indices):

            image, label = dataset[idx]


            image_to_pred = image.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)

            with torch.no_grad():
                output = model(image_to_pred)

                prob_vector = torch.softmax(output, dim=1)
                pred_p, pred_class = prob_vector.topk(1, dim=1)
                

                pred_class = pred_class.item()
                pred_p = pred_p.item()

                correct_prediction = pred_class == label.item()


            title_color = 'green' if correct_prediction else 'red'
            axes[i].imshow(undo_normalize(image), cmap='gray')
            axes[i].set_title(f"{classes[pred_class]} ({pred_p*100: .1f}%)", color=title_color)
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
    else:

        test_indices = np.random.randint(0, len(dataset), size=num_images)

        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

        for i, idx in enumerate(test_indices):

            image, label = dataset[idx]

            axes[i].imshow(undo_normalize(image), cmap='gray')
            axes[i].set_title(f"Label: {classes[label]}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        
        
def plot_loss_accuracy(train_loss_list, val_loss_list, train_acc_list, val_acc_list):

    num_epochs = len(train_loss_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(np.arange(1, num_epochs+1), train_loss_list, color="green", label='Train Loss')
    ax1.plot(np.arange(1, num_epochs+1), val_loss_list, color="red", label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.legend()

    ax2.plot(np.arange(1, num_epochs+1), train_acc_list, color="green", label='Train Accuracy')
    ax2.plot(np.arange(1, num_epochs+1), val_acc_list, color="red", label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
