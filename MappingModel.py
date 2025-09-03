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
from SamOutVXP.model3 import SamOut
from SamOutVXP.high_vocab3 import UniVoc
import csv
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 常量定义 =====
# 模型参数
LATENT_DIM = 512
TEXT_EMBED_DIM = 512
HIDDEN_DIM = 1024
NUM_LAYERS = 8
DROPOUT_RATE = 0.2
# 统一词表尺寸（训练早期脚本使用 voc_size+1 保存模型）
VOC_SIZE = UniVoc().voc_size + 1
# 新增: 标签最大 token 数 (用于序列监督而非平均池化)
LABEL_MAX_TOKENS = 8

# 训练参数
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 1e-5

# 路径参数
VAE_MODEL_PATH = './ckpts/imageVAE/image_vae_best_model.pth'
LM_MODEL_PATH = './SamOutVXP/model_pretrain_cap_1_sft_new.pth'
IMAGE_ROOT_DIR = './data/Fruit-Images-Dataset'  # 图像根目录（其下每个子文件夹名即文本描述）
SAVE_DIR = './mapping_model'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 新增: 序列化映射网络(Transformer) 取代简单平均池化监督 ==========
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
        # x: (B,L,D)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class MappingNetwork(nn.Module):
    """
    视觉潜变量 -> 文本 token 序列嵌入预测。
    步骤:
      1) latent 线性映射到 hidden
      2) 复制扩展到 LABEL_MAX_TOKENS 序列长度
      3) 加入可学习位置 & 查询参数 (pos + query)
      4) 多层 Transformer 细化
      5) 输出 (B, LABEL_MAX_TOKENS, TEXT_EMBED_DIM)
    损失: 与真实 token 嵌入序列做 mask MSE。
    """
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
        # latent: (B, LATENT_DIM)
        b = latent.size(0)
        h = self.input_proj(latent)  # (B,H)
        # 扩展为序列: (B,L,H)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        q = self.query_embed.unsqueeze(0).repeat(b, 1, 1)
        p = self.pos_embed.unsqueeze(0)
        x = h + q + p
        key_padding_mask = None  # 这里不对预测端做mask
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.out_norm(x)
        x = self.out_proj(x)  # (B,L,output_dim)
        return x

# ========== 修改数据集: 返回 token 序列嵌入 + mask =========
class ImageTextDataset(Dataset):
    """基于目录结构的图像-文本对数据集。
    目录结构: root_dir/文本描述/图片文件
    文本描述 = 子文件夹名称
    """
    def __init__(self, image_root_dir, vae_model, lm_model, tokenizer, transform=None, exts=(".jpg", ".png", ".jpeg", ".bmp")):
        self.image_root_dir = image_root_dir
        self.vae_model = vae_model
        self.lm_model = lm_model
        self.transform = transform
        self.tokenizer = tokenizer
        self.exts = set(e.lower() for e in exts)
        self.samples = []
        for sub in os.listdir(image_root_dir):
            sub_dir = os.path.join(image_root_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            text = sub
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
            image_latent = image.unsqueeze(0).to(device)
            mean, logvar = self.vae_model.encode(image_latent)
            image_latent = self.vae_model.reparameterize(mean, logvar).squeeze(0).cpu()
        # 文本 -> token 序列嵌入
        token_ids = self.text_to_token_ids(text)
        # 截断 / 填充
        token_ids = token_ids[:LABEL_MAX_TOKENS]
        length = len(token_ids)
        if length < LABEL_MAX_TOKENS:
            token_ids = token_ids + [0] * (LABEL_MAX_TOKENS - length)
        token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            seq_embed = self.lm_model.em(token_tensor).squeeze(0).cpu()  # (L, D)
        # mask
        mask = torch.zeros(LABEL_MAX_TOKENS, dtype=torch.float32)
        mask[:length] = 1.0
        return image_latent, seq_embed, mask, token_ids

    def text_to_token_ids(self, text):
        return self.tokenizer.encode(text)

# ========== 预处理: 保存序列嵌入与 mask =========
def create_dataset(image_root_dir, vae_model, lm_model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageTextDataset(image_root_dir=image_root_dir, vae_model=vae_model, lm_model=lm_model, tokenizer=UniVoc(), transform=transform)
    os.makedirs(SAVE_DIR, exist_ok=True)
    data_path = os.path.join(SAVE_DIR, 'preprocessed_seq_data.pkl')
    if os.path.exists(data_path):
        ans = input(f"检测到已存在序列预处理数据 {data_path}，是否加载？(y/[n]): ")
        if ans.lower()=='y':
            print(f'加载已有序列预处理数据: {data_path}')
            return data_path
        else:
            print('重新生成序列预处理数据...')
    img_latents, txt_seq_embeds, masks = [], [], []
    token_ids_list = []  # 新增
    label_token_ids = set()
    for i in tqdm(range(len(dataset)), desc='预处理数据(序列)'):
        il, se, mk, _tok_ids = dataset[i]  # 修正: 原来只解包3个
        img_latents.append(il)
        txt_seq_embeds.append(se)
        masks.append(mk)
        text = dataset.samples[i][1]
        raw_ids_full = dataset.text_to_token_ids(text)
        raw_ids_trunc = raw_ids_full[:LABEL_MAX_TOKENS]
        length = len(raw_ids_trunc)
        padded = raw_ids_trunc + [0]*(LABEL_MAX_TOKENS-length)
        token_ids_list.append(torch.tensor(padded, dtype=torch.long))
        for tid in raw_ids_trunc:
            label_token_ids.add(tid)
    data = {
        'image_latents': torch.stack(img_latents),            # (N, LATENT)
        'text_seq_embeddings': torch.stack(txt_seq_embeds),   # (N, L, D)
        'masks': torch.stack(masks),                          # (N, L)
        'token_ids': torch.stack(token_ids_list)              # (N, L)
    }
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
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
        # 新增: token_ids
        if 'token_ids' not in data:
            raise ValueError('预处理文件缺少 token_ids，请删除旧的 preprocessed_seq_data.pkl 并重新生成。')
        self.token_ids = data['token_ids']  # (N, L)
    def __len__(self):
        return len(self.image_latents)
    def __getitem__(self, idx):
        return self.image_latents[idx], self.text_seq_embeddings[idx], self.masks[idx], self.token_ids[idx]

# ========== 序列损失(掩码 MSE) =========
def sequence_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2  # (B,L,D)
    diff = diff.mean(dim=-1)     # (B,L)
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

# ========== 新增: 余弦与混合序列损失 =========
def sequence_cosine_loss(pred, target, mask, eps=1e-8):
    # pred/target: (B,L,D)
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    cos = (pred_n * targ_n).sum(dim=-1)  # (B,L)
    loss = (1.0 - cos) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)

def sequence_cosine_hybrid_loss(pred, target, mask, alpha=0.1):
    cos_part = sequence_cosine_loss(pred, target, mask)
    pred_norm = pred.norm(dim=-1)
    targ_norm = target.norm(dim=-1)
    norm_part = ((pred_norm - targ_norm) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
    return cos_part + alpha * norm_part

# ========== 训练函数修改 =========
def train_mapping_model():
    vae_model = ImageVAE(latent_dim=LATENT_DIM).to(device)
    vae_model.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae_model.eval()
    lm_model = SamOut(voc_size=VOC_SIZE, hidden_size=TEXT_EMBED_DIM, num_heads=8, num_layers=8).to(device)
    lm_model.load_state_dict(torch.load(LM_MODEL_PATH, map_location=device))
    lm_model.eval()
    vocab = UniVoc()
    data_path = os.path.join(SAVE_DIR, 'preprocessed_seq_data.pkl')
    if os.path.exists(data_path):
        ans = input(f"检测到已存在序列预处理数据 {data_path}，是否加载？(y/[n]): ")
        if not ans.lower().startswith('y'):
            print('重新生成序列预处理数据...')
            data_path = create_dataset(IMAGE_ROOT_DIR, vae_model, lm_model)
        else:
            print(f'加载已有序列预处理数据: {data_path}')
    else:
        data_path = create_dataset(IMAGE_ROOT_DIR, vae_model, lm_model)
    dataset = PreprocessedDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    mapping_model = MappingNetwork(
        input_dim=LATENT_DIM,
        output_dim=TEXT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,  # 可调更大
        dropout_rate=DROPOUT_RATE,
        seq_len=LABEL_MAX_TOKENS,
        heads=8,
        mlp_ratio=4.0
    ).to(device)

    optimizer = optim.AdamW(mapping_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_csv_path = os.path.join(SAVE_DIR, 'train_log_seq.csv')
    with open(log_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch','train_loss','val_loss','lr'])

    best_val = float('inf')
    train_losses, val_losses = [] , []
    # 新增: 选择损失类型: 'mse' | 'cos' | 'cos_hybrid'
    LOSS_TYPE = 'cos_hybrid'
    def loss_fn(pred, target, mask):
        if LOSS_TYPE == 'mse':
            return sequence_mse_loss(pred, target, mask)
        if LOSS_TYPE == 'cos':
            return sequence_cosine_loss(pred, target, mask)
        if LOSS_TYPE == 'cos_hybrid':
            return sequence_cosine_hybrid_loss(pred, target, mask, alpha=0.1)
        raise ValueError(f'Unknown LOSS_TYPE: {LOSS_TYPE}')

    for epoch in range(NUM_EPOCHS):
        mapping_model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [训练]'):
            img_lat, txt_seq, msk, _tok_ids = batch
            img_lat = img_lat.to(device)
            txt_seq = txt_seq.to(device)
            msk = msk.to(device)
            optimizer.zero_grad()
            pred_seq = mapping_model(img_lat)  # (B,L,D)
            loss = loss_fn(pred_seq, txt_seq, msk)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * img_lat.size(0)
        tr_loss /= len(train_loader.dataset)
        train_losses.append(tr_loss)
        mapping_model.eval()
        vl_loss = 0.0
        # 评估阶段，显示部分生成序列
        show_eval = True
        shown = False
        with torch.no_grad():
            emb_weight = lm_model.em.weight.data  # (V,D)
            emb_norm = F.normalize(emb_weight, dim=-1).cpu()
            for batch in val_loader:
                img_lat, txt_seq, msk, tok_ids = batch
                img_lat = img_lat.to(device)
                txt_seq = txt_seq.to(device)
                msk = msk.to(device)
                tok_ids = tok_ids.to(device)
                pred_seq = mapping_model(img_lat)
                loss = loss_fn(pred_seq, txt_seq, msk)
                vl_loss += loss.item() * img_lat.size(0)
                if show_eval and not shown:
                    for i in range(min(3, img_lat.size(0))):
                        length = int(msk[i].sum().item())
                        # 预测 token ids
                        pred = pred_seq[i].cpu()
                        pred_norm = F.normalize(pred, dim=-1)
                        pred_ids = torch.matmul(pred_norm, emb_norm.t()).argmax(dim=-1).tolist()
                        pred_ids = pred_ids[:length]
                        # 真实 token ids
                        true_ids = tok_ids[i][:length].cpu().tolist()
                        print(f'[Eval Sample {i}]')
                        print('  Pred:', vocab.decode(pred_ids))
                        print('  True:', vocab.decode(true_ids))
                    shown = True
        vl_loss /= len(val_loader.dataset)
        val_losses.append(vl_loss)
        scheduler.step(vl_loss)
        lr_cur = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | 训练损失: {tr_loss:.6f} | 验证损失: {vl_loss:.6f} | lr: {lr_cur:.3e}')
        with open(log_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch+1,tr_loss,vl_loss,lr_cur])
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

# ===== 主函数 =====
if __name__ == "__main__":
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 训练映射模型
    train_mapping_model()