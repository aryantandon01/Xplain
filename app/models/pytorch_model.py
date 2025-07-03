import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # binary classification
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    model = SimpleNet()
    # for now just random weights; later load from file if saved
    model.eval()
    return model
