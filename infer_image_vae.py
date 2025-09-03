import torch
import argparse
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from BasicVAE.ImageVAE import ImageVAE

def infer_image_vae(image_path, model_path, output_dir, latent_dim=512, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = ImageVAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    # 推理
    with torch.no_grad():
        recon_img, _, _ = model(image_tensor)
    # 反归一化
    recon_img = recon_img.squeeze(0).cpu()
    recon_img = recon_img * 0.5 + 0.5
    recon_img = recon_img.clamp(0, 1)
    # 保存原图和重建图
    os.makedirs(output_dir, exist_ok=True)
    recon_save_path = os.path.join(output_dir, 'reconstructed.png')
    plt.imsave(recon_save_path, recon_img.permute(1, 2, 0).numpy())
    print(f"重建图已保存到: {recon_save_path}")

if __name__ == "__main__":
    image_path = "tmpDBA4.png"
    model_path = "./ckpts/imageVAE/image_vae_best_model.pth"
    output_dir = "./vae_infer_output"
    latent_dim = 512

    infer_image_vae(image_path=image_path, model_path=model_path, output_dir=output_dir, latent_dim=latent_dim)