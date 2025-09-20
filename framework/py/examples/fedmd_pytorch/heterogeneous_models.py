import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCifarNet(nn.Module):
    """작은 CNN 모델 - 클라이언트 1용"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.p = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))
        x = self.p(F.relu(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MediumCifarNet(nn.Module):
    """중간 크기 CNN 모델 - 클라이언트 2용"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.p = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*4*4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))
        x = self.p(F.relu(self.c2(x)))
        x = self.p(F.relu(self.c3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)

class LargeCifarNet(nn.Module):
    """큰 CNN 모델 - 클라이언트 3용"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.c4 = nn.Conv2d(256, 512, 3, padding=1)
        self.p = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512*2*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))
        x = self.p(F.relu(self.c2(x)))
        x = self.p(F.relu(self.c3(x)))
        x = self.p(F.relu(self.c4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.fc4(x)

class ResNetLikeCifarNet(nn.Module):
    """ResNet 스타일 모델 - 클라이언트 4용 (선택사항)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

def get_model_for_client(client_id: int, num_classes: int = 10):
    """클라이언트 ID에 따라 다른 모델 반환"""
    if client_id == 0:
        return SmallCifarNet(num_classes)
    elif client_id == 1:
        return MediumCifarNet(num_classes)
    elif client_id == 2:
        return LargeCifarNet(num_classes)
    else:
        return ResNetLikeCifarNet(num_classes)

def print_model_info(model, client_id):
    """모델 정보 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_name = model.__class__.__name__
    print(f"Client {client_id + 1} Model: {model_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print()
