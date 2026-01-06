
# model.py
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetAttrClassifier(nn.Module):
    """
    Multi-label classifier for 40 CelebA attributes with various backbone options.

    Handles normalization conversion: input is expected in [-1, 1] range (DiffAE format),
    and is converted to ImageNet normalization for pretrained backbones.
    """
    def __init__(self, backbone='resnet50', num_classes=40, pretrained=True, dropout=0.5):
        super().__init__()

        # Register ImageNet normalization parameters as buffers
        # These move with the model across devices
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Use new weights API (torchvision >= 0.13)
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.resnet18(weights=weights)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.resnet50(weights=weights)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.resnet101(weights=weights)
            feat_dim = self.net.fc.in_features
            self.net.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.efficientnet_b0(weights=weights)
            feat_dim = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
        elif backbone == 'efficientnet_b1':
            weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.efficientnet_b1(weights=weights)
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
        # Convert from [-1, 1] (DiffAE) to [0, 1]
        x = (x + 1) / 2
        # Apply ImageNet normalization
        x = (x - self.imagenet_mean) / self.imagenet_std

        features = self.net(x)
        return self.classifier(features)
