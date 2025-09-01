import torch
from model import MultiModalVAE
import matplotlib.pyplot as plt
import yaml
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

config = yaml.safe_load(open('config.yaml', 'r', encoding='utf-8'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = MultiModalVAE(n_mfcc=config['n_mfcc'],
                      audio_dim=config['audio_dim'],
                       text_dim=config['text_dim'],
                       image_dim=config['image_dim'],
                       image_size=config['image_size']).to(device=device)
model.load_state_dict(torch.load('ckpts/final_model.pth', map_location=device))
model.eval()


with torch.no_grad():
    # 随机采样潜在向量
    z = torch.randn(1,256 * 3).to(device)
    audio_gen, text_gen, image_gen = model.decoder(z)
    print("Generated audio shape:", audio_gen.shape)
    print("Generated text shape:", text_gen.shape)
    print("Generated image shape:", image_gen.shape)

    # 保存生成的图像
    image_gen = image_gen.squeeze(0).cpu()   # 反归一化
    print(image_gen)
    plt.imsave('generated_image.png', image_gen.permute(1, 2, 0).numpy())
    print("Generated image saved as 'generated_image.png'")
    # 保存生成的文本
    generated_text = model.decoder.text_decoder.tokenizer.decode(torch.argmax(text_gen, dim=-1).squeeze().cpu().numpy(), skip_special_tokens=True)
    with open('generated_text.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print("Generated text saved as 'generated_text.txt'")
    # 保存生成的音频
    from scipy.io.wavfile import write
    audio_gen = audio_gen.squeeze(0).squeeze(0).cpu().numpy()
    write('generated_audio.wav', config['sampling_rate'], audio_gen)
    print("Generated audio saved as 'generated_audio.wav'")