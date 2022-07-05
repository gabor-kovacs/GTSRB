import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from dataset import GTSRB, split_train_val
from model import Net
from loss import FocalLoss
from test import test


USE_AUGMENTATION = True
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOG_INTERVAL = 10
EPOCHS = 20
GAMMA = 5.0

n_classes = 43

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, val = split_train_val(ratio=0.2)
    train_dataset = GTSRB(train, use_augmentation=True)
    val_dataset = GTSRB(val, use_augmentation=True)
    train_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=False)

    model = Net()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # weights for loss_fn
    weights_np = np.arange(n_classes, dtype=float)
    for i in range(n_classes):
        weights_np[i] = len(train) / n_classes / len(train[train['ClassId'] == i])
    weights = torch.from_numpy(weights_np).float()
    weights = weights.to(device)

    loss_fn = FocalLoss(alpha=weights, gamma=GAMMA, reduction="mean")


    def train(epoch):
        model.train()
        for batch_idx, sample in enumerate(train_loader):
            data, labels = sample
            data = data.to(device) # why is this not an in place operation like model.to(device)??
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print(f"Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.1f}%)] Loss: {loss.data.item():.5f}")

    def validate():
        model.eval()
        val_loss = 0
        correct = 0
        for data, labels in val_loader:
            data = data.to(device) 
            labels = labels.to(device)
            output = model(data)
            val_loss += loss_fn(output, labels).data.item()
            pred = output.data.max(1, keepdim=True)[1]  
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        val_loss /= len(val_loader.dataset)
        print(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100.0 * correct / len(val_loader.dataset):.2f}%)")

        return val_loss

    best_val_loss = 9999

    for epoch in range(EPOCHS):
        train(epoch)
        val_loss = validate()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(Path(__file__).parent.absolute(), "out", f"model_{epoch + 1}.pth"))
            torch.save(model.state_dict(), Path(Path(__file__).parent.absolute(), "out", "model_best.pth"))
            print("Saved best model")

    test()

if __name__ == "__main__":
    main()