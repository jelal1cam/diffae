
# model.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetAttrClassifier(nn.Module):
    """
    Multi-label classifier for 40 CelebA attributes with various backbone options.
    """
    def __init__(self, backbone='resnet50', num_classes=40, pretrained=True, dropout=0.5):
        super().__init__()
        
        if backbone == 'resnet18':
            self.net = models.resnet18(pretrained=pretrained)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.net = models.resnet50(pretrained=pretrained)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'resnet101':
            self.net = models.resnet101(pretrained=pretrained)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.net = models.efficientnet_b0(pretrained=pretrained)
            feat_dim = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
        elif backbone == 'efficientnet_b1':
            self.net = models.efficientnet_b1(pretrained=pretrained)
            feat_dim = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone {backbone}")
        
        # Multi-label classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.net(x)
        return self.classifier(features)
