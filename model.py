import torch
import torch.nn as nn
import torchinfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CatDogCNN(nn.Module):
    def __init__(self, imsize: tuple[int, int] = (128, 128)):
        self.imsize = imsize
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=512 * 2 * 2, out_features=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def summary(self) -> None:
        torchinfo.summary(
            self, input_size=[1, 3, self.imsize[0], self.imsize[1]], depth=2
        )
