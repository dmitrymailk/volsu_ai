import torch
from torch import nn


class TestModel(nn.Module):
    def __init__(self, n_classes=None):
        super(TestModel, self).__init__()

        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(28 * 28, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x
