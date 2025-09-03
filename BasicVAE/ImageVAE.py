import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split  # 不再需要随机划分

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class ImageVAE(nn.Module):
    """
    一个适用于图像的变分自编码器 (VAE) 模型。
    输入 [batch_size, 3, 128, 128] 的张量。
    """
    def __init__(self, latent_dim=256):
        super(ImageVAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器 Encoder - 输出均值和方差
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        # 计算卷积层输出的特征维度
        self.conv_output_dim = 256 * 8 * 8 # 根据网络结构计算得出

        # 映射到潜在空间的均值和对数方差
        self.fc_mean = nn.Linear(self.conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_dim, latent_dim)

        # 解码器 Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """重参数化技巧：从N(mean, std)中采样，同时允许梯度回溯"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) # 从标准正态分布中采样随机噪声
            z = mean + eps * std
            return z
        else:
            # 推理时直接使用均值，减少随机性
            return mean

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mean, logvar
    

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp")):
        self.transform = transform
        self.image_label_pairs = []
        self.exts = {e.lower() for e in exts}
        # root_dir 为包含类别子文件夹的目录
        if not os.path.isdir(root_dir):
            raise RuntimeError(f'Dataset root dir not found: {root_dir}')
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for file in os.listdir(label_dir):
                if os.path.splitext(file)[1].lower() in self.exts:
                    self.image_label_pairs.append((os.path.join(label_dir, file), label))
        if len(self.image_label_pairs) == 0:
            raise RuntimeError(f'No images found under {root_dir}')

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        from PIL import Image
        image_path, label = self.image_label_pairs[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def vae_loss(recon_x, x, mean, logvar, recon_weight=1.0, kl_weight=1.0):
    """
    VAE损失函数 = 重建损失 + KL散度损失

    参数:
        recon_x: 模型重建的输出
        x: 原始输入数据
        mean: 潜在空间分布的均值
        logvar: 潜在空间分布的对数方差
        recon_weight: 重建损失的权重
        kl_weight: KL散度损失的权重
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    total_loss = recon_weight * recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

# 新增: 与 MappingModelwithR1 一致的数据集路径
FRUITS100_ROOT = './data/fruits100'
TRAIN_IMAGE_DIR = os.path.join(FRUITS100_ROOT, 'train')
VAL_IMAGE_DIR = os.path.join(FRUITS100_ROOT, 'val')
VAE_CKPT_DIR = './ckpts/imageVAE'
os.makedirs(VAE_CKPT_DIR, exist_ok=True)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not (os.path.isdir(TRAIN_IMAGE_DIR) and os.path.isdir(VAL_IMAGE_DIR)):
        raise RuntimeError(f'Expected train/val dirs at {TRAIN_IMAGE_DIR} and {VAL_IMAGE_DIR}')
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 直接使用显式 train / val 目录
    train_dataset = ImageDataset(root_dir=TRAIN_IMAGE_DIR, transform=image_transform)
    val_dataset = ImageDataset(root_dir=VAL_IMAGE_DIR, transform=image_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = ImageVAE(latent_dim=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model = model.to(device)

    epochs = 50
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_path = os.path.join(VAE_CKPT_DIR, 'image_vae_best_model.pth')

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch, _ in train_loader:
            imgs = batch.to(device)
            optimizer.zero_grad()
            recon_imgs, mean, logvar = model(imgs)
            loss, recon_loss, kl_loss = vae_loss(recon_x=recon_imgs, x=imgs, mean=mean, logvar=logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader) / len(batch)
        loss_history.append(avg_loss)

        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for val_batch, _ in val_loader:
                val_imgs = val_batch.to(device)
                recon_imgs, mean, logvar = model(val_imgs)
                val_loss, _, _ = vae_loss(recon_x=recon_imgs, x=val_imgs, mean=mean, logvar=logvar)
                val_total_loss += val_loss.item()
        avg_val_loss = val_total_loss / len(val_loader) / len(val_batch)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(VAE_CKPT_DIR, f'image_vae_epoch_{epoch+1}.pth'))

    plt.figure()
    plt.plot(range(1, epochs+1), loss_history, marker='o', label='Train Loss')
    plt.plot(range(1, epochs+1), val_loss_history, marker='s', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('VAE Training & Validation Loss Curve (fruits100)')
    plt.legend()
    plt.grid(True)
    plt.show()