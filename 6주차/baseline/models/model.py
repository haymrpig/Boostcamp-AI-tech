import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

_all_models=["resnet18", "mnasnet1_0", "wide_resnet50_2", "resnext50_32x4d", "mobilenet_v2",
            "googlenet", "inception_v3", "densenet161", "squeezenet1_0", "vgg16",
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b3_pruned', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6',
            'efficientnet_b7', 'efficientnet_b8']

class resnet18(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnet18(pretrained = self.pretrained)
        self.in_features = self.model.fc.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class mnasnet1_0(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.mnasnet1_0(pretrained = self.pretrained)
        self.in_features = self.model.classifier[1].out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class wide_resnet50_2(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.wide_resnet50_2(pretrained = self.pretrained)
        self.in_features = self.model.fc.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class resnext50_32x4d(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.resnext50_32x4d(pretrained = self.pretrained)
        self.in_features = self.model.fc.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class mobilenet_v2(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.mobilenet_v2(pretrained = self.pretrained)
        self.in_features = self.model.classifier[1].out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class googlenet(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.googlenet(pretrained = self.pretrained)
        self.in_features = self.model.fc.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class inception_v3(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.inception_v3(pretrained = self.pretrained)
        self.in_features = self.model.fc.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x


class densenet161(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.densenet161(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x


class squeezenet1_0(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.squeezenet1_0(pretrained = self.pretrained)
        self.in_features = self.model.classifier[1].in_channels
        self.num_classes = num_classes
        self.model.classifier[1] = nn.Conv2d(self.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x


class vgg16(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.vgg16(pretrained = self.pretrained)
        self.in_features = self.model.classifier[6].out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )
        
    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class squeezenet1_0(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.squeezenet1_0(pretrained = self.pretrained)
        self.in_features = self.model.classifier[1].in_channels
        self.num_classes = num_classes
        self.model.classifier[1] = nn.Conv2d(self.in_features, num_classes)
        

    def forward(self, x):
        x = self.model(x)
        return x


class efficientnet_b0(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b0(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b1(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b1(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b2(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b2(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b3(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b3(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b3_pruned(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b3_pruned(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x


class efficientnet_b4(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b4(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b5(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b5(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b6(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b6(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b7(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b7(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

class efficientnet_b8(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, freeze=False):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.efficientnet_b8(pretrained = self.pretrained)
        self.in_features = self.model.classifier.out_features
        self.num_classes = num_classes
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.in_features, 512),
            nn.ReLU(inplace=True), 
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x