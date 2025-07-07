import torch
import torch.nn as nn
from torchvision.models import resnet18
from models.m_utils import test_model, train_model, get_predictions


class ResNet18(nn.Module):
    def __init__(self, dataset, n_classes, device):
        super(ResNet18, self).__init__()

        self.device = device
        self.resnet = resnet18()

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)
        if dataset == "fmnist":
            # FashionMNIST has 1 channel (grayscale) and 10 classes
            self.resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )  # 1 channel

    def forward(self, x):
        # Forward pass through the ResNet18 model
        return self.resnet(x)

    def train_model(self, global_model, trainloader, true_label_mapping, epochs, mu):
        train_model(self, global_model, trainloader, true_label_mapping, epochs, mu, self.device)

    def test_model(self, testloader, true_label_mapping):
        return test_model(self, testloader, true_label_mapping, self.device)

    def get_predictions(self, testloader, true_label_mapping):
        return get_predictions(self, testloader, true_label_mapping, self.device)