import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Define a simple CNN for the reverse diffusion process


class DiffusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.fc = nn.Linear(32 * 7 * 7, 784)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))  # Use sigmoid to ensure output is in [0,1]
        x = x.view(-1, 1, 28, 28)  # Reshape back to image dimensions
        return x


# Beta schedule for the diffusion process
T = 1000  # Total number of diffusion steps
beta_start = 1e-4
beta_end = 2e-2
betas = torch.linspace(beta_start, beta_end, T)

# Simulate the diffusion process


def diffusion(x0, betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    noises = torch.randn_like(x0, device=x0.device) * torch.sqrt(1. - alphas_cumprod[-1])
    xT = torch.sqrt(alphas_cumprod[-1]) * x0 + noises
    return xT, noises

# Attempt to reverse the diffusion process


def reverse_diffusion(model, xT, betas, device):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    for t in reversed(range(0, T)):
        with torch.no_grad():
            xT = (xT - (1 - alphas[t]) / torch.sqrt(1 - alphas_cumprod[t]) * model(xT)) / torch.sqrt(alphas[t])
    return xT


# Training loop setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
epochs = 5
for epoch in range(epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        xT, noise = diffusion(data, betas.to(device))
        x_reconstructed = reverse_diffusion(model, xT, betas.to(device), device)
        loss = criterion(x_reconstructed, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item()}')
