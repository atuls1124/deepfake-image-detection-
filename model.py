# model.py - Definition of the DeepfakeDetector model
import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # Replace the final classifier
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_ftrs, 2)  # 2 output classes: real and fake
        )
        
    def forward(self, x):
        return self.efficientnet(x)