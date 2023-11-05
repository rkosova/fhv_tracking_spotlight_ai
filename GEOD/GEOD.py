import torch

from torch import nn


class GEOD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.feature_extraction_layers = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=(4, 4), stride=2), # out => 248x248
            nn.MaxPool2d((2, 2), stride=2), # out => 124x124
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 192, kernel_size=(3, 3)), # out => 122x122
            nn.MaxPool2d((2, 2), stride=2), # out => 61
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.classification_layers = nn.Sequential(
            nn.Linear(192*61*61, 16),
            nn.Sigmoid()
        )

    
    def forward(self, X):
        features = self.feature_extraction_layers(X)
        features = self.flatten(features) 
        classification = self.classification_layers(features)
        return classification
