import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer
import csv
from helpers import BertTextSimilarity
import librosa

# 初始化tokenizer及相关参数
TOKENIZER_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
vocab_size = tokenizer.vocab_size

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, num_layers=2, out_dim=256):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.contiguous()
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class ImageEncoder(nn.Module):
    def __init__(self, image_size, out_dim):
        super(ImageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 使用输入参数指定flatten后的维度
        self.flatten_dim = (image_size // 8) * (image_size // 8) * 128
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)    

class TextEncoder(nn.Module):
    def __init__(self, model='./bert-base-uncased', latent_dim=64, freeze_bert=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.bert = AutoModel.from_pretrained(model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_output_dim = self.bert.config.hidden_size
        self.fc_mu = nn.Linear(self.bert_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.bert_output_dim, latent_dim)
    def forward(self, text):
        # text: list of strings
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        sentence_rep = outputs.last_hidden_state[:, 0, :] # [CLS] token
        mu = self.fc_mu(sentence_rep)
        logvar = self.fc_logvar(sentence_rep)
        return mu, logvar

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2, eos_token_id=2, max_len=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.eos_token_id = eos_token_id
        self.max_len = max_len

    def forward(self, z, target_seq=None, teacher_forcing=True):
        batch_size = z.size(0)
        decode_len = target_seq.size(1) if target_seq is not None else self.max_len
        hidden = self.latent_to_hidden(z).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        logits_seq = []
        inputs = torch.full((batch_size, 1), self.eos_token_id, dtype=torch.long, device=z.device)
        for t in range(decode_len):
            embedded = self.embedding(inputs)
            out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            logits = self.fc_out(out.squeeze(1))
            logits_seq.append(logits.unsqueeze(1))
            if teacher_forcing and target_seq is not None and t < target_seq.size(1):
                inputs = target_seq[:, t].unsqueeze(1)
            else:
                inputs = logits.argmax(dim=1, keepdim=True)
            if not teacher_forcing and (inputs == self.eos_token_id).all():
                break
        return torch.cat(logits_seq, dim=1)  # (batch, decode_len, vocab_size)
    
class AudioDecoder(nn.Module):
    def __init__(self, latent_dim, feature_dim, hidden_dim=256, num_layers=2, max_len=100, eos_value=None):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
        self.latent_to_input = nn.Linear(latent_dim, feature_dim)
        self.max_len = max_len
        self.eos_value = eos_value

    def forward(self, z, target_seq=None, teacher_forcing=True):
        input_step = self.latent_to_input(z).unsqueeze(1)  # (batch, 1, feature_dim)
        outputs = []
        hidden = None
        for t in range(self.max_len):
            out, hidden = self.lstm(input_step, hidden)
            next_feat = self.fc_out(out.squeeze(1))
            outputs.append(next_feat.unsqueeze(1))
            if teacher_forcing and target_seq is not None and t < target_seq.size(1):
                input_step = target_seq[:, t].unsqueeze(1)
            else:
                input_step = next_feat.unsqueeze(1)
            if not teacher_forcing and self.eos_value is not None:
                if (next_feat.abs().max(dim=1)[0] < self.eos_value).all():
                    break
        return torch.cat(outputs, dim=1)

class ImageDecoder(nn.Module):
    def __init__(self, hidden_dim, image_size=128):
        super(ImageDecoder, self).__init__()
        self.image_size = image_size
        # 计算初始特征图大小
        start_size = image_size // 16  # 128 -> 8, 64 -> 4
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, start_size * start_size * 128),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.fc(z)
        batch_size = out.size(0)
        start_size = self.image_size // 16
        out = out.view(batch_size, 128, start_size, start_size)
        out = self.deconv(out)
        return out  # (batch, 3, image_size, image_size)

class MultiModalEncoder(nn.Module):
    def __init__(self, audio_encoder, text_encoder, image_encoder, audio_dim, text_dim, image_dim):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.audio_fc_mu = nn.Linear(audio_dim, audio_dim)
        self.audio_fc_logvar = nn.Linear(audio_dim, audio_dim)
        self.text_fc_mu = nn.Linear(text_dim*2, text_dim)
        self.text_fc_logvar = nn.Linear(text_dim*2, text_dim)
        self.image_fc_mu = nn.Linear(image_dim, image_dim)
        self.image_fc_logvar = nn.Linear(image_dim, image_dim)

    def forward(self, audio, audio_caption, image, image_caption):
        audio_feat = self.audio_encoder(audio)
        mu_text_audio, logvar_text_audio = self.text_encoder(audio_caption)
        mu_text_image, logvar_text_image = self.text_encoder(image_caption)
        text_feat = torch.cat([mu_text_audio, mu_text_image], dim=1)
        image_feat = self.image_encoder(image)
        mu_audio = self.audio_fc_mu(audio_feat)
        logvar_audio = self.audio_fc_logvar(audio_feat)
        mu_text = self.text_fc_mu(text_feat)
        logvar_text = self.text_fc_logvar(text_feat)
        mu_image = self.image_fc_mu(image_feat)
        logvar_image = self.image_fc_logvar(image_feat)
        return mu_audio, logvar_audio, mu_text, logvar_text, mu_image, logvar_image
    
class MultiModalDecoder(nn.Module):
    def __init__(self, audio_latent_dim, text_latent_dim, image_latent_dim, audio_decoder, text_decoder, image_decoder):
        super(MultiModalDecoder, self).__init__()
        self.audio_latent_dim = audio_latent_dim
        self.text_latent_dim = text_latent_dim
        self.image_latent_dim = image_latent_dim
        self.audio_decoder = audio_decoder
        self.text_decoder = text_decoder
        self.image_decoder = image_decoder

    def forward(self, z, target_seq=None):
        audio_z = z[:, :self.audio_latent_dim]
        text_z = z[:, self.audio_latent_dim:self.audio_latent_dim+self.text_latent_dim]
        image_z = z[:, self.audio_latent_dim+self.text_latent_dim:]
        audio_recon = self.audio_decoder(audio_z)
        text_recon = self.text_decoder(text_z, target_seq=target_seq)
        image_recon = self.image_decoder(image_z)
        return audio_recon, text_recon, image_recon    

class MultiModalVAE(nn.Module):
    def __init__(self,n_mfcc, audio_dim, text_dim, image_dim, image_size,):
        super(MultiModalVAE, self).__init__()
        audio_encoder = AudioEncoder(input_dim=n_mfcc, out_dim=audio_dim)
        text_encoder = TextEncoder(latent_dim=256)
        text_decoder = TextDecoder(
            latent_dim=256,
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=256,
            num_layers=2,
            eos_token_id=2,
            max_len=50
        )
        image_encoder = ImageEncoder(image_size=image_size, out_dim=image_dim)
        audio_decoder = AudioDecoder(latent_dim=audio_dim, feature_dim=n_mfcc, hidden_dim=audio_dim)
        image_decoder = ImageDecoder(hidden_dim=image_dim, image_size=image_size)
        self.encoder = MultiModalEncoder(audio_encoder=audio_encoder, text_encoder=text_encoder, image_encoder=image_encoder, audio_dim=audio_dim, text_dim=text_dim, image_dim=image_dim)
        self.decoder = MultiModalDecoder(audio_latent_dim=audio_dim, text_latent_dim=text_dim, image_latent_dim=image_dim, audio_decoder=audio_decoder, text_decoder=text_decoder, image_decoder=image_decoder)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, audio, audio_caption, image, image_caption, text_tokens=None):
        mu_audio, logvar_audio, mu_text, logvar_text, mu_image, logvar_image = self.encoder(audio, audio_caption, image, image_caption)
        z_audio = self.reparameterize(mu_audio, logvar_audio)
        z_text = self.reparameterize(mu_text, logvar_text)
        z_image = self.reparameterize(mu_image, logvar_image)
        z = torch.cat([z_audio, z_text, z_image], dim=1)
        audio_recon, text_recon, image_recon = self.decoder(z, target_seq=text_tokens)
        return audio_recon, text_recon, image_recon, z_audio, z_text, z_image, mu_audio, logvar_audio, mu_text, logvar_text, mu_image, logvar_image
    
    
# 定义损失函数
class MultiModalVAELoss():
    def __init__(self):
        self.bert_sim = BertTextSimilarity()

    def calculate(
            self,
            audio_recon,
            audio_latent,
            audio,
            text_recon,
            text_latent,
            text,
            image_recon,
            image_latent,
            image,
            mu_audio,
            logvar_audio,
            mu_text,
            logvar_text,
            mu_image,
            logvar_image,
            audio_captions,
            image_captions,
            device,
            kl_weight=1.0,
            audio_image_align_weight=1.0,
            audio_weight=1.0,
            text_weight=1.0,
            image_weight=1.0,
            pad_token_id=0
        ):
        # 重构损失
        audio_recon_loss = F.mse_loss(audio_recon, audio, reduction='mean') * audio_weight

        if text_recon.dim() == 3:  # (batch, seq_len, vocab_size)
            text_recon_loss = F.cross_entropy(
                text_recon.view(-1, text_recon.size(-1)),
                text.view(-1),
                ignore_index=pad_token_id,
                reduction='mean'
            ) * text_weight
        else:
            raise ValueError("text_recon 应为 (batch, seq_len, vocab_size) 的 logits，但得到的维度为 {}".format(text_recon.dim()))

        image_recon_loss = F.mse_loss(image_recon, image, reduction='mean') * image_weight

        # KL散度（分别计算三个模态）
        kl_audio = -0.5 * torch.sum(1 + logvar_audio - mu_audio.pow(2) - logvar_audio.exp())
        kl_text = -0.5 * torch.sum(1 + logvar_text - mu_text.pow(2) - logvar_text.exp())
        kl_image = -0.5 * torch.sum(1 + logvar_image - mu_image.pow(2) - logvar_image.exp())
        kl_loss = kl_audio + kl_text + kl_image
        kl_loss = kl_loss * kl_weight

        # 跨模态对齐损失
        sim_audio_image = self.bert_sim.calculate(audio_captions, image_captions).to(device)
        latent_sim_audio_image = F.cosine_similarity(audio_latent, image_latent, dim=1).to(device)
        audio_image_alignment_loss = F.mse_loss(latent_sim_audio_image, sim_audio_image) * audio_image_align_weight

        total_loss = (
            audio_recon_loss +
            text_recon_loss +
            image_recon_loss +
            kl_loss +
            audio_image_alignment_loss 
        )
        return total_loss, audio_recon_loss, text_recon_loss, image_recon_loss, kl_loss, audio_image_alignment_loss

class MultiModalDataset(Dataset):
    def __init__(self, image_dir, image_captions_file, audio_dir, audio_captions_file, sampling_rate = 44100, image_transform=None, audio_transform=None, n_mfcc = 100, max_text_len=50, max_audio_len=100):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.image_captions = {}
        self.audio_dir = audio_dir
        self.audio_transform = audio_transform
        self.audio_captions = {}
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        with open(image_captions_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for line in reader:
                img = line[0]
                caption = line[1]
                if img not in self.image_captions:
                    self.image_captions[img] = []
                self.image_captions[img].append(caption)
        with open(audio_captions_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.audio_captions[row[0]] = row[1:]

        self.image_files = list(self.image_captions.keys())
        self.audio_files = list(self.audio_captions.keys())
        self.image_usage_count = {img: 0 for img in self.image_files}
        self.audio_usage_count = {aud: 0 for aud in self.audio_files}

        self.min_audio_length = np.inf
        for aud_name in self.audio_files:
            audio_path = os.path.join(self.audio_dir, aud_name.strip())
            waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)
            self.min_audio_length = min(self.min_audio_length, len(waveform))

    def __len__(self):
        return max(len(self.image_files), len(self.audio_files))
    def __getitem__(self, idx):
        # 图像-文字对
        img_idx = idx % len(self.image_files)
        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.tensor(np.array(image)).float().view(-1) / 255.0
        image_caption = self.image_captions[img_name][0]
        self.image_usage_count[img_name] += 1
        image_usage = self.image_usage_count[img_name]
        # 音频-文字对
        aud_idx = idx % len(self.audio_files)
        aud_name = self.audio_files[aud_idx]
        audio_path = os.path.join(self.audio_dir, aud_name.strip())
        waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)
        waveform = waveform[:int(self.min_audio_length)]
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc)
        audio_seq = torch.tensor(mfcc.T, dtype=torch.float32)
        # padding/truncation
        if audio_seq.size(0) < self.max_audio_len:
            pad = torch.zeros(self.max_audio_len-audio_seq.size(0), audio_seq.size(1))
            audio_seq = torch.cat([audio_seq, pad], dim=0)
        else:
            audio_seq = audio_seq[:self.max_audio_len]
        audio_len = min(audio_seq.size(0), self.max_audio_len)
        if self.audio_transform:
            audio_seq = self.audio_transform(audio_seq)
        else:
            audio_seq = (audio_seq - audio_seq.mean()) / (audio_seq.std() + 1e-9)
        audio_caption = self.audio_captions[aud_name][0] if self.audio_captions[aud_name] else ''
        self.audio_usage_count[aud_name] += 1
        audio_usage = self.audio_usage_count[aud_name]
        return image, image_usage, audio_seq, audio_len, audio_caption, audio_usage, image_caption