import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 128
epochs = 10
raw_dim = 784
hidden_dim = 400  # 第一层的维度，
latent_dim = 20  # 隐变量的维度
# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# 转换后的张量的形状为(C, H, W)，其中C是通道数，H是高度，W是宽度。
# 这一步骤也自动将图像的像素值从[0, 255]缩放到[0.0, 1.0]之间。
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Variational Autoencoder Definition


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 网络的维度是 raw_dim -> hidden_dim -> latent_dim -> hidden_dim -> raw_dim
        # encoder raw_dim -> hidden_dim -> latent_dim
        # decoder latent_dim -> hidden_dim -> raw_dim

        self.encoder_l1 = nn.Linear(raw_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)  # Mean μ
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance σ^2
        self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_l2 = nn.Linear(hidden_dim, raw_dim)

    def encode(self, x):
        h = F.relu(self.encoder_l1(x))
        return self.encoder_mu(h), self.encoder_logvar(h)

    def sample_z(self, mu, logvar):
        std_var = torch.exp(0.5 * logvar)  # 标准差
        eps = torch.randn_like(std_var)  # 生成一个与std相同大小的张量，值是从标准正态分布中随机抽取的
        return mu + eps * std_var

    def decode(self, z):
        h = F.relu(self.decoder_l1(z))
        return torch.sigmoid(self.decoder_l2(h))  # 归一化

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, raw_dim))
        z = self.sample_z(mu, logvar)
        return self.decode(z), mu, logvar


# Model, Optimizer, and Loss Function
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    # 计算重建的误差，只和recon_x, x 有关
    RECON = F.mse_loss(recon_x, x.view(-1, raw_dim), reduction='sum')
    # 计算KL散度 D(P(Z|X)||Q(Z)) 只和 mu, logvar 有关
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return RECON + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader.dataset)}')


for epoch in range(1, epochs + 1):
    train(epoch)

# Save a sample of generated images
with torch.no_grad():
    sample = torch.randn(64, latent_dim).to(device)
    sample = model.decode(sample).cpu()
    save_image(sample.view(64, 1, 28, 28), 'sample_' + str(epoch) + '.png')
