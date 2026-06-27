import torch
import torch.nn as nn
import torch.optim as optim

from src.config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from src.models.cnn import CropDiseaseCNN
from src.data.dataloader import get_dataloaders
from src.training.trainer import train_one_epoch


def train(train_dir, valid_dir):

    train_loader, valid_loader = get_dataloaders(
        train_dir,
        valid_dir,
        BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CropDiseaseCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    return model