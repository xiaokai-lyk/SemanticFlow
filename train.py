import torch
from torch.utils.data import DataLoader
import yaml
from model import MultiModalVAE, MultiModalDataset
from model import MultiModalVAELoss
import matplotlib.pyplot as plt
import time
import numpy as np
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
import os
import csv

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# 日志记录函数
def log_training(epoch, batch_idx, num_batches, loss, batch_time):
    print(f"Epoch {epoch+1} Batch {batch_idx+1}/{num_batches} | Loss: {loss:.4f} | Time: {batch_time:.2f}s")

# 可视化函数，支持滑动平均和实时更新
def plot_losses(losses_dict, save_path=None, window=20, pause_time=0.01):
    plt.clf()
    plt.figure(1)
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
        if len(losses) >= window:
            smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(losses)), smooth, label=f"Smoothed {label} (window={window})")
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.pause(pause_time)

def plot_latent_space(z_audio, z_text, z_image, step, save_dir='latent_vis'):
    os.makedirs(save_dir, exist_ok=True)
    z_audio_np = z_audio.detach().cpu().numpy()
    z_text_np = z_text.detach().cpu().numpy()
    z_image_np = z_image.detach().cpu().numpy()
    pca = PCA(n_components=2)
    z_audio_2d = pca.fit_transform(z_audio_np)
    z_text_2d = pca.fit_transform(z_text_np)
    z_image_2d = pca.fit_transform(z_image_np)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.scatter(z_audio_2d[:,0], z_audio_2d[:,1], alpha=0.5)
    plt.title('z_audio')
    plt.subplot(1,3,2)
    plt.scatter(z_text_2d[:,0], z_text_2d[:,1], alpha=0.5)
    plt.title('z_text')
    plt.subplot(1,3,3)
    plt.scatter(z_image_2d[:,0], z_image_2d[:,1], alpha=0.5)
    plt.title('z_image')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_step_{step}.png')

# 训练函数
def train_vae(model, dataloader, optimizer, epochs, device, vae_loss_fn,save_every, save_path, scheduler=None ):
    model.train()
    all_losses = {
        'Total Loss': [],
        'Audio Recon Loss': [],
        'Text Recon Loss': [],
        'Image Recon Loss': [],
        'KL Loss': [],
        'Audio-Image Alignment Loss': []
    }
    plt.ion()  # 打开交互模式
    global_step = 0
    # 日志文件初始化
    log_file = 'train_log.csv'
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'epoch', 'batch_idx', 'total_loss', 'audio_recon_loss', 'text_recon_loss', 'image_recon_loss', 'kl_loss', 'audio_image_alignment_loss'])
    for epoch in range(epochs):
        if save_every and save_path is not None and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"{save_path}/epoch{epoch+1}.pth")
        total_loss = 0
        start_epoch = time.time()
        for batch_idx, (image, image_usage, audio_seq, audio_len, audio_caption, audio_usage, image_caption) in enumerate(dataloader):
            batch_start = time.time()
            image = image.to(device)
            audio_seq = audio_seq.to(device)
            # 文本token化
            audio_caption_tokens = model.encoder.text_encoder.tokenizer(
                audio_caption, return_tensors='pt', padding=True, truncation=True, max_length=50
            )['input_ids'].to(device)
            image_caption_tokens = model.encoder.text_encoder.tokenizer(
                image_caption, return_tensors='pt', padding=True, truncation=True, max_length=50
            )['input_ids'].to(device)
            # 保证 image_caption_tokens 长度一致
            image_caption_tokens = model.encoder.text_encoder.tokenizer(
                image_caption, return_tensors='pt', padding='max_length', truncation=True, max_length=50
            )['input_ids'].to(device)

            # VAE forward
            audio_recon, text_recon, image_recon, z_audio, z_text, z_image, mu_audio, logvar_audio, mu_text, logvar_text, mu_image, logvar_image = model(
                audio_seq, audio_caption, image, image_caption, text_tokens=image_caption_tokens
            )

            # KL Annealing
            kl_weight = min(config['kl_max_weight'], global_step / config['kl_anneal_steps'])
            # 损失计算
            loss, audio_recon_loss, text_recon_loss, image_recon_loss, kl_loss, audio_image_alignment_loss = vae_loss_fn.calculate(
                audio_recon=audio_recon,
                audio_latent=z_audio,
                audio=audio_seq,
                text_recon=text_recon,
                text_latent=z_text,
                text=image_caption_tokens,
                image_recon=image_recon,
                image_latent=z_image,
                image=image,
                mu_audio=mu_audio,
                logvar_audio=logvar_audio,
                mu_text=mu_text,
                logvar_text=logvar_text,
                mu_image=mu_image,
                logvar_image=logvar_image,
                audio_captions=audio_caption,
                image_captions=image_caption,
                pad_token_id=pad_token_id,
                device=device,
                kl_weight=kl_weight,
                audio_image_align_weight=config['audio_image_align_weight']
            )
            loss.backward()

            # Apply gradient clipping if enabled
            if config['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

            optimizer.step()
            batch_time = time.time() - batch_start
            log_training(epoch, batch_idx, len(dataloader), loss.item(), batch_time)
            total_loss += loss.item()

            # Track individual losses
            all_losses['Total Loss'].append(loss.item())
            all_losses['Audio Recon Loss'].append(audio_recon_loss.item())
            all_losses['Text Recon Loss'].append(text_recon_loss.item())
            all_losses['Image Recon Loss'].append(image_recon_loss.item())
            all_losses['KL Loss'].append(kl_loss.item())
            all_losses['Audio-Image Alignment Loss'].append(audio_image_alignment_loss.item())

            # 保存日志到csv
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, epoch, batch_idx,
                    loss.item(), audio_recon_loss.item(), text_recon_loss.item(), image_recon_loss.item(), kl_loss.item(), audio_image_alignment_loss.item()
                ])
            # 实时更新loss曲线
            plot_losses(all_losses, save_path=None, window=20, pause_time=0.01)
            # 潜在空间可视化
            if config.get('save_lalent_every', -1) > 0 and (global_step + 1) % config['save_lalent_every'] == 0:
                plot_latent_space(z_audio, z_text, z_image, global_step)
            global_step += 1
        if scheduler:
            scheduler.step()
        # 显示数据使用次数
        print(f"Image usage counts (sample): {image_usage}")
        print(f"Audio usage counts (sample): {audio_usage}")
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} completed in {time.time()-start_epoch:.2f}s, Avg Loss: {avg_loss:.4f}")
    # 训练结束后保存最终loss曲线
    plot_losses(all_losses, save_path='train_loss.png', window=20, pause_time=0.01)
    plt.ioff()
    if save_path is not None:
        torch.save(model.state_dict(), f"{save_path}/final_model.pth")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 主程序
if __name__ == "__main__":
    # 设置超参数
    config = load_config('config.yaml')
    audio_dim = config['audio_dim']
    text_dim = config['text_dim']
    image_dim = config['image_dim']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = MultiModalVAE(n_mfcc=config['n_mfcc'],audio_dim=audio_dim, text_dim=text_dim, image_dim=image_dim,image_size = config['image_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if config.get('use_scheduler', False):
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    else:
        scheduler = None
    vae_loss_fn = MultiModalVAELoss()

    # 定义pad_token_id
    pad_token_id = config.get('pad_token_id', 0)
    

    image_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    dataset = MultiModalDataset(
        image_dir=config['train_image_path'],
        image_captions_file=config['train_image_captions'],
        audio_dir=config['train_audio_path'],
        audio_captions_file=config['train_audio_captions'],
        sampling_rate=config['sampling_rate'],
        image_transform=image_transform,
        n_mfcc=config['n_mfcc']
        )
    num_workers = config.get('num_workers', 0)
    if num_workers == -1:
        num_workers = os.cpu_count()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=config.get('pin_memory', False), num_workers=num_workers, persistent_workers=config.get('persistent_workers', False))
    
    # 训练模型
    train_vae(model, dataloader, optimizer, epochs, device, vae_loss_fn, scheduler=scheduler, save_every=config.get('save_every', -1), save_path=config.get('save_path', None))