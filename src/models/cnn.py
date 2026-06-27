import torch
import torch.nn as nn


class CropDiseaseCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 38)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x