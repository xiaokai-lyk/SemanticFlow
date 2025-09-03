import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import pickle
from BasicVAE.ImageVAE import ImageVAE
import csv
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 随机种子 ==========
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常量定义 =====
LATENT_DIM = 512
HIDDEN_DIM = 1024
NUM_LAYERS = 8
DROPOUT_RATE = 0.2
LABEL_MAX_TOKENS = 8
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
WEIGHT_DECAY = 1e-5

# 模型路径
VAE_MODEL_PATH = './ckpts/imageVAE/image_vae_best_model.pth'
DEEPSEEK_MODEL_PATH = './DeepSeek-R1-Distill-Qwen-1.5B'
# 旧: IMAGE_ROOT_DIR = './data/Fruit-Images-Dataset'
# 新数据集（包含 train / val 两个子目录，各目录下再按类别文件夹存放）
FRUITS100_ROOT = './data/fruits100'
TRAIN_IMAGE_DIR = os.path.join(FRUITS100_ROOT, 'train')
VAL_IMAGE_DIR = os.path.join(FRUITS100_ROOT, 'val')
SAVE_DIR = './mapping_model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Transformer映射网络 ==========
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class MappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.1, seq_len=LABEL_MAX_TOKENS, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_embed = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
    def forward(self, latent):
        b = latent.size(0)
        h = self.input_proj(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        q = self.query_embed.unsqueeze(0).repeat(b, 1, 1)
        p = self.pos_embed.unsqueeze(0)
        x = h + q + p
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        return self.out_proj(x)

# ========== 数据集 ==========
class ImageTextDataset(Dataset):
    """目录结构: root_dir/类名/图片; 类名即文本标签（下划线转空格）"""
    def __init__(self, image_root_dir, vae_model, lm_model, tokenizer, transform=None, exts=(".jpg", ".png", ".jpeg", ".bmp")):
        self.image_root_dir = image_root_dir
        self.vae_model = vae_model
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.transform = transform
        self.exts = set(e.lower() for e in exts)
        self.samples = []
        for sub in os.listdir(image_root_dir):
            sub_dir = os.path.join(image_root_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            text = sub.replace('_', ' ')
            for fname in os.listdir(sub_dir):
                if os.path.splitext(fname)[1].lower() in self.exts:
                    self.samples.append((os.path.join(sub_dir, fname), text))
        if len(self.samples) == 0:
            raise RuntimeError(f"在目录 {image_root_dir} 下未找到任何图像。")
        self.vae_model.eval(); self.lm_model.eval()
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        with torch.no_grad():
            lat_in = image.unsqueeze(0).to(device)
            mean, logvar = self.vae_model.encode(lat_in)
            image_latent = self.vae_model.reparameterize(mean, logvar).squeeze(0).cpu()
        # 文本 -> token ids
        token_ids = self.text_to_token_ids(text)
        token_ids = token_ids[:LABEL_MAX_TOKENS]
        length = len(token_ids)
        pad_id = self.tokenizer.pad_token_id
        if length < LABEL_MAX_TOKENS:
            token_ids = token_ids + [pad_id]*(LABEL_MAX_TOKENS-length)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            seq_embed = self.lm_model.get_input_embeddings()(token_tensor).squeeze(0).cpu()  # (L, D)
        mask = torch.zeros(LABEL_MAX_TOKENS, dtype=torch.float32)
        mask[:length] = 1.0
        return image_latent, seq_embed, mask, token_ids
    def text_to_token_ids(self, text):
        # 不添加特殊token，按标签原文切分
        return self.tokenizer.encode(text, add_special_tokens=False)

# ========== 预处理 ==========
def create_dataset(image_root_dir, vae_model, lm_model, tokenizer, out_name='preprocessed_seq_data.pkl', save_label_tokens=True):
    """
    针对单一根目录(例如 train 或 val)，按类别子文件夹收集样本并生成（或加载）预处理文件。
    out_name: 生成的 pkl 文件名
    save_label_tokens: 是否保存 label_token_set.pkl（只在训练集上保存一次以免覆盖）
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = ImageTextDataset(image_root_dir=image_root_dir, vae_model=vae_model, lm_model=lm_model, tokenizer=tokenizer, transform=transform)
    os.makedirs(SAVE_DIR, exist_ok=True)
    data_path = os.path.join(SAVE_DIR, out_name)
    if os.path.exists(data_path):
        ans = input(f"检测到已存在预处理数据 {data_path}，是否加载？(y/[n]): ")
        if ans.lower()=='y':
            print(f'加载已有预处理数据: {data_path}')
            return data_path
        else:
            print('重新生成预处理数据...')
    img_latents, txt_seq_embeds, masks, token_ids_list = [], [], [], []
    label_token_ids = set()
    for i in tqdm(range(len(dataset)), desc=f'预处理数据(序列): {os.path.basename(image_root_dir)}'):
        il, se, mk, _tok_ids = dataset[i]
        img_latents.append(il)
        txt_seq_embeds.append(se)
        masks.append(mk)
        token_ids_list.append(torch.tensor(_tok_ids, dtype=torch.long))
        for tid in _tok_ids:
            if tid != tokenizer.pad_token_id:
                label_token_ids.add(tid)
    data = {
        'image_latents': torch.stack(img_latents),
        'text_seq_embeddings': torch.stack(txt_seq_embeds),
        'masks': torch.stack(masks),
        'token_ids': torch.stack(token_ids_list)
    }
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    if save_label_tokens:
        with open(os.path.join(SAVE_DIR, 'label_token_set.pkl'), 'wb') as f:
            pickle.dump(sorted(list(label_token_ids)), f)
        print(f"已保存标签 token 集合，大小: {len(label_token_ids)}")
    return data_path

class PreprocessedDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.image_latents = data['image_latents']
        self.text_seq_embeddings = data['text_seq_embeddings']
        self.masks = data['masks']
        if 'token_ids' not in data:
            raise ValueError('预处理文件缺少 token_ids，需要重新生成。')
        self.token_ids = data['token_ids']
    def __len__(self):
        return len(self.image_latents)
    def __getitem__(self, idx):
        return self.image_latents[idx], self.text_seq_embeddings[idx], self.masks[idx], self.token_ids[idx]

# ========== 损失函数 ==========
def sequence_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)

def sequence_cosine_loss(pred, target, mask):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    cos = (pred_n * targ_n).sum(dim=-1)
    loss = (1.0 - cos) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)

def sequence_cosine_hybrid_loss(pred, target, mask, alpha=0.1):
    cos_part = sequence_cosine_loss(pred, target, mask)
    pred_norm = pred.norm(dim=-1)
    targ_norm = target.norm(dim=-1)
    norm_part = ((pred_norm - targ_norm)**2 * mask).sum() / mask.sum().clamp_min(1.0)
    return cos_part + alpha * norm_part

# ========== 训练主流程 ==========
def train_mapping_model():
    # 加载VAE
    vae_model = ImageVAE(latent_dim=LATENT_DIM).to(device)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae_model.eval()
    # 加载语言模型
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_PATH).to(device)
    lm_model.eval()
    embed_dim = lm_model.get_input_embeddings().embedding_dim

    # 使用显式 train / val 目录，不再随机划分
    if not (os.path.isdir(TRAIN_IMAGE_DIR) and os.path.isdir(VAL_IMAGE_DIR)):
        raise RuntimeError(f'期望目录 {TRAIN_IMAGE_DIR} 与 {VAL_IMAGE_DIR} 存在且包含类别子文件夹。')

    train_data_path = create_dataset(TRAIN_IMAGE_DIR, vae_model, lm_model, tokenizer, out_name='preprocessed_seq_train.pkl', save_label_tokens=True)
    val_data_path = create_dataset(VAL_IMAGE_DIR, vae_model, lm_model, tokenizer, out_name='preprocessed_seq_val.pkl', save_label_tokens=False)

    train_dataset = PreprocessedDataset(train_data_path)
    val_dataset = PreprocessedDataset(val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mapping_model = MappingNetwork(
        input_dim=LATENT_DIM,
        output_dim=embed_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        seq_len=LABEL_MAX_TOKENS,
        heads=8,
        mlp_ratio=4.0
    ).to(device)

    optimizer = optim.AdamW(mapping_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_csv_path = os.path.join(SAVE_DIR, 'train_log_seq.csv')
    with open(log_csv_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(['epoch','train_loss','val_loss','lr'])

    best_val = float('inf')
    train_losses, val_losses = [], []
    LOSS_TYPE = 'cos_hybrid'
    def loss_fn(pred, target, mask):
        if LOSS_TYPE == 'mse':
            return sequence_mse_loss(pred, target, mask)
        if LOSS_TYPE == 'cos':
            return sequence_cosine_loss(pred, target, mask)
        if LOSS_TYPE == 'cos_hybrid':
            return sequence_cosine_hybrid_loss(pred, target, mask, alpha=0.1)
        raise ValueError('Unknown LOSS_TYPE')

    emb_weight = lm_model.get_input_embeddings().weight.data
    emb_norm_all = F.normalize(emb_weight, dim=-1).cpu()

    for epoch in range(NUM_EPOCHS):
        mapping_model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [训练]'):
            img_lat, txt_seq, msk, _tok_ids = batch
            img_lat = img_lat.to(device)
            txt_seq = txt_seq.to(device)
            msk = msk.to(device)
            optimizer.zero_grad()
            pred_seq = mapping_model(img_lat)
            loss = loss_fn(pred_seq, txt_seq, msk)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * img_lat.size(0)
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)

        mapping_model.eval()
        vl_loss = 0.0
        shown = False
        with torch.no_grad():
            for batch in val_loader:
                img_lat, txt_seq, msk, tok_ids = batch
                img_lat = img_lat.to(device)
                txt_seq = txt_seq.to(device)
                msk = msk.to(device)
                tok_ids = tok_ids.to(device)
                pred_seq = mapping_model(img_lat)
                loss = loss_fn(pred_seq, txt_seq, msk)
                vl_loss += loss.item() * img_lat.size(0)
                if not shown:
                    for i in range(min(3, img_lat.size(0))):
                        length = int(msk[i].sum().item())
                        pred = pred_seq[i].cpu()[:length]
                        pred_norm = F.normalize(pred, dim=-1)
                        sim = torch.matmul(pred_norm, emb_norm_all.t())
                        pred_ids = sim.argmax(dim=-1).tolist()
                        true_ids = tok_ids[i][:length].cpu().tolist()
                        print(f'[Eval Sample {i}]')
                        print('  Pred:', tokenizer.decode(pred_ids, skip_special_tokens=True))
                        print('  True:', tokenizer.decode(true_ids, skip_special_tokens=True))
                    shown = True
        vl_loss /= len(val_loader.dataset)
        val_losses.append(vl_loss)
        scheduler.step(vl_loss)
        lr_cur = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | 训练损失: {tr_loss:.6f} | 验证损失: {vl_loss:.6f} | lr: {lr_cur:.3e}')
        with open(log_csv_path, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([epoch+1, tr_loss, vl_loss, lr_cur])
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(mapping_model.state_dict(), os.path.join(SAVE_DIR, 'best_mapping_model_seq.pth'))
        if (epoch+1) % 10 == 0:
            torch.save(mapping_model.state_dict(), os.path.join(SAVE_DIR, f'mapping_model_seq_epoch_{epoch+1}.pth'))

    plt.figure()
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='训练损失')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='验证损失')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('序列映射模型损失曲线'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, 'loss_curve_seq.png'))
    plt.show()

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_mapping_model()