import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.layer = nn.Sequential(
            nn.Linear(1*150*75, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.shape[0], -1)
        x = self.layer(x)

        return x


