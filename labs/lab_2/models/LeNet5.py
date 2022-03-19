import torch.nn.functional as F
import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.hardtanh = nn.Hardtanh(-2, 2)  # good results
        self.relu6 = nn.ReLU6()  # good results
        self.celu = nn.CELU()  # good results

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1),
            # nn.BatchNorm2d(10),
            self.hardtanh,
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, stride=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(16),
            self.hardtanh,
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=3, stride=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(120),
            self.celu,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=120 * 4, out_features=84),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(84),
            self.relu6,
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
