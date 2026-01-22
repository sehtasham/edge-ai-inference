import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

