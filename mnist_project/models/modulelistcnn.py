import torch.nn as nn

class ModuleListCNN(nn.Module):
    def __init__(self):
        super(ModuleListCNN, self).__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        ])
        self.fc_blocks = nn.ModuleList([
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        ])

    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_blocks:
            x = layer(x)
        return x