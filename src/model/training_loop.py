
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid 
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.model.model_classes import EfficientNet_ContextualData


def log_metrics_tensor_board(data_type, accuracy, top_5_accuracy, loss, epoch ):
    writer = SummaryWriter()
    writer.add_scalar(f"Loss -  {data_type}", loss, epoch)
    writer.add_scalar(f"Accuracy -  {data_type}", 100 * accuracy, epoch)
    writer.add_scalar(f"Top 5 Accuracy - {data_type}", 100 * top_5_accuracy, epoch)
    return


def save_model(model, criterion, optimizer, epoch, model_name):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"runs/model_{model_name}.pth")
    return


def run_epoch(model, loader, criterion, optimizer=None, epoch=0, mode="train"):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    top_5_correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):  # Only compute gradients during training
        for images, contextual_data, labels in loader:
            images, contextual_data, labels = images.to(device), contextual_data.to(device), labels.to(device)
            outputs = model(images, contextual_data)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            # count total correct predicitions in batch
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # count total matches of label in top 5 class predictions 
            _, top_5_predicted = outputs.topk(5, dim=1)
            top_5_correct += torch.isin(labels, top_5_predicted).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    top_5_accuracy = top_5_correct / total
    log_metrics_tensor_board(mode, accuracy, top_5_accuracy, avg_loss, epoch)

    print(f"{mode.capitalize()} Accuracy: {100 * accuracy:.2f}%")
    print(f"Top 5 {mode.capitalize()} Accuracy: {100 * top_5_accuracy:.2f}%")
    print(f"Epoch {epoch+1} - {mode.capitalize()} Loss: {avg_loss:.4f}")
    return avg_loss, accuracy


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, epochs=5, early_stop_limit=5):
    best_val_loss = float('inf')
    early_stop_count = 0
    # lists to track of loss and accuracy metrics
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, epoch, mode="train")
        # if no val loader, evaluate stopping based on train metrics (e.g. when training on validation after training on train before evaluating on test)
        if val_loader is None:
            val_loss, val_acc = train_loss, train_acc
        else:
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, epoch, mode="validation")
        # store metrics
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # save best model so far
            save_model(model, criterion, optimizer,  epoch, model_name)
        else:
            early_stop_count += 1
        if early_stop_count == early_stop_limit:
            break
    eval_metrics = {
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list
        }
    return eval_metrics


def full_train_pipeline(model, model_name, train_loader, val_loader, device):
    criterion= nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    eval_metrics = train_model(model, model_name, train_loader, val_loader, criterion, optimizer, epochs=5, early_stop_limit=5)
    # load best model after early stopping
    checkpoint = torch.load(model_name + ".pth")
    model = EfficientNet_ContextualData(1486, 3)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # train for 5 more epochs on validation set to get best performance
    train_model(model, model_name, val_loader, None, criterion, optimizer, epochs=5, early_stop_limit=3)
    # load and return final model
    checkpoint = torch.load(model_name + ".pth")
    model = EfficientNet_ContextualData(1486, 3)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, eval_metrics


def test_model(model, test_loader, device):
    criterion= nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, epoch=0, mode="validation")
    return test_loss, test_acc