import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from SamOutVXP.model3 import SamOut
from SamOutVXP.high_vocab3 import UniVoc
from BasicVAE.ImageVAE import ImageVAE
from MappingModel import MappingNetwork, LATENT_DIM, TEXT_EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE, LM_MODEL_PATH, VAE_MODEL_PATH, SAVE_DIR
import pickle
import torch.nn as nn

"""
映射推理脚本 (image -> latent -> mapped text embedding -> 初始 token -> 语言模型生成)
============================================================
核心流程:
1. 加载组件:
   - 图像VAE: 提取图像潜变量 z (维度 LATENT_DIM)
   - 映射模型 MappingNetwork: 将 z 映射到文本嵌入空间 e_txt_pred (维度 TEXT_EMBED_DIM)
   - 语言模型 SamOut: 通过词嵌入矩阵 (embedding table) 与 e_txt_pred 做相似度搜索获取初始 token 引导生成。
2. 初始 token 选择:
   - 计算预测向量与所有词嵌入的余弦相似度，选 Top-K 作为候选。
   - 选择 top1 作为起始 token，或保留多个 token 作为前缀（可选）。
3. 文本生成:
   - 使用改进采样: 温度 / top-k / top-p / 重复惩罚。
   - 逐步追加 token，直到达到长度或遇到 eos_token。
4. 输出:
   - 返回生成文本及调试信息（可选）。

注意事项:
- 当前映射模型只回归平均 token 嵌入（训练阶段使用 mean pooling），属于“全句级”压缩，信息损失较大，因此生成文本更偏语义提示 / 类别描述，而非精确 caption。
- 可改进方向: 预测多 token 序列嵌入、引入对比损失、使用 CLIP 风格双塔结构等。
"""

# 默认路径 (可根据训练脚本实际保存位置调整)
VAE_MODEL_PATH = './ckpts/imageVAE/image_vae_best_model.pth'
LM_MODEL_PATH = './SamOutVXP/model_pretrain_cap_6.pth'
MAPPING_MODEL_PATH = './mapping_model/mapping_model_seq_epoch_10.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# ------------------------- 采样函数 -------------------------

def sample_with_repetition_suppression(logits: torch.Tensor,
                                       generated_ids: list,
                                       temperature: float = 1.0,
                                       top_k: int = None,
                                       top_p: float = None,
                                       repetition_penalty: float = 1.2) -> int:
    # 重复惩罚
    if generated_ids and repetition_penalty > 1.0:
        recent_ids = set(generated_ids[-min(64, len(generated_ids)):])
        for token_id in recent_ids:
            if token_id < logits.size(0):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
    # 温度
    logits = logits / max(temperature, 1e-6)
    if temperature < 1e-6:
        return logits.argmax().item()
    # 数值安全
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e8, neginf=-1e8)
    probs = torch.softmax(logits, dim=-1)
    # top-k
    if top_k is not None and top_k > 0:
        top_k = min(top_k, probs.size(-1))
        topk_probs, topk_idx = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs)
        mask.scatter_(-1, topk_idx, topk_probs)
        probs = mask
    # top-p
    if top_p is not None and 0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        csum = torch.cumsum(sp, dim=-1)
        remove = csum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sp[remove] = 0
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, si, sp)
        probs = filtered
    # 归一化
    s = probs.sum()
    if s <= 0:
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs = probs / s
    probs = probs + 1e-10 / probs.numel()
    probs = probs / probs.sum()
    return torch.multinomial(probs.cpu(), 1).item()

# ------------------------- 图像处理 -------------------------

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    return _transform(img).unsqueeze(0).to(device)

# ------------------------- 初始化 / 加载 -------------------------

def load_models():
    vocab = UniVoc()
    ckpt = torch.load(LM_MODEL_PATH, map_location=device)
    emb_w = ckpt['em.weight'] if 'em.weight' in ckpt else ckpt['state_dict']['em.weight']
    ckpt_vocab_size = emb_w.shape[0]
    lm = SamOut(
        voc_size=ckpt_vocab_size,
        hidden_size=TEXT_EMBED_DIM,
        num_heads=8,
        num_layers=8
    ).to(device)
    lm.load_state_dict(ckpt, strict=True)
    lm.eval()
    vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae.eval()
    mapping = MappingNetwork(
        input_dim=LATENT_DIM,
        output_dim=TEXT_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    mapping.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device))
    mapping.eval()
    # 读取标签 token 集合，若不存在则为 None
    label_token_set_path = os.path.join(SAVE_DIR, 'label_token_set.pkl')
    allowed_ids = None
    if os.path.exists(label_token_set_path):
        with open(label_token_set_path, 'rb') as f:
            allowed_ids = set(pickle.load(f))
        print(f"[Info] 载入标签 token 集合, 大小: {len(allowed_ids)}")
    else:
        print('[Warn] 未找到标签 token 集合文件, 不进行前缀候选约束。')
    return vocab, lm, vae, mapping, allowed_ids

# ------------------------- 由映射向量得到初始 token -------------------------

def embedding_to_start_tokens(pred_embed: torch.Tensor, lm: SamOut, top_k: int = 5, allowed_ids=None):
    emb_matrix = lm.em.weight.data  # (V, D)
    pred = F.normalize(pred_embed.unsqueeze(0), dim=-1)
    emb_norm = F.normalize(emb_matrix, dim=-1)
    sims = torch.matmul(pred, emb_norm.t()).squeeze(0)
    if allowed_ids is not None:
        mask = torch.full_like(sims, -1e9)
        idx = torch.tensor(list(allowed_ids), dtype=torch.long, device=sims.device)
        mask[idx] = sims[idx]
        sims = mask
    topk = torch.topk(sims, k=min(top_k, (sims > -1e9/2).sum().item()))
    return topk.indices.tolist(), topk.values.tolist()

# ============== 新增: 多 token 前缀生成 ==============
SPECIAL_TOKEN_IDS = {0, 1, 2, 3, 5, 6}  # 根据你的词表可再调整

def multi_token_prefix(pred_embed: torch.Tensor,
                       lm: SamOut,
                       prefix_len: int = 4,
                       mode: str = 'residual',
                       candidate_k: int = 40,
                       temperature: float = 0.7,
                       avoid_set=None,
                       allow_repetition: bool = False,
                       allowed_ids=None):
    """
    将单一映射向量扩展为多个前缀 token。
    若 pred_embed 形状为 (L,D) (当前映射网络输出序列)，自动做 mean pooling 得到 (D)。
    mode:
      residual  : 迭代选择最相似 token, 每次减去其(归一化)向量的成分 (贪心稀疏分解)
      topk_sample: 每步从 top-K 相似中按 softmax(相似/temperature) 采样
      hybrid    : 先 residual 选2个, 再 topk_sample 选剩余
    candidate_k: 每步截断的 top-K 候选容量。
    temperature: 采样温度 (仅对 topk_sample/hybrid 后半部分生效)
    avoid_set  : 要避开的 token id 集合
    allow_repetition: 是否允许重复 token
    """
    # 兼容序列输出 (L,D)
    if pred_embed.dim() == 2:  # (L,D)
        pred_embed = pred_embed.mean(dim=0)
    emb = lm.em.weight.data  # (V,D)
    emb_norm = F.normalize(emb, dim=-1)
    vec = F.normalize(pred_embed, dim=0)
    V = emb_norm.size(0)
    if avoid_set is None:
        avoid_set = SPECIAL_TOKEN_IDS
    chosen = []
    used = set()
    allowed_tensor = None
    if allowed_ids is not None:
        allowed_tensor = torch.tensor(sorted(list(allowed_ids)), device=emb_norm.device)

    def apply_masks(sims_masked):
        if allowed_tensor is not None:
            mask_all = torch.full_like(sims_masked, -1e9)
            mask_all[allowed_tensor] = sims_masked[allowed_tensor]
            sims_masked = mask_all
        for tid in (avoid_set if not allow_repetition else avoid_set - set()):
            if 0 <= tid < V:
                sims_masked[tid] = -1e9
        if not allow_repetition:
            for tid in used:
                sims_masked[tid] = -1e9
        return sims_masked

    def pick_residual_step(cur_vec):
        sims = torch.matmul(cur_vec, emb_norm.t())
        sims_masked = apply_masks(sims.clone())
        topv, topi = torch.topk(sims_masked, k=1)
        return topi.item(), topv.item(), sims

    def pick_topk_sample(cur_vec):
        sims = torch.matmul(cur_vec, emb_norm.t())
        sims_masked = apply_masks(sims.clone())
        valid_count = (sims_masked > -1e9/2).sum().item()
        k = min(candidate_k, valid_count if valid_count > 0 else 1)
        topv, topi = torch.topk(sims_masked, k=k)
        logits = topv / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sel_local = torch.multinomial(probs, 1).item()
        return topi[sel_local].item(), probs[sel_local].item(), sims

    if mode == 'residual':
        steps_plan = ['residual'] * prefix_len
    elif mode == 'topk_sample':
        steps_plan = ['topk'] * prefix_len
    elif mode == 'hybrid':
        r = max(1, min(2, prefix_len - 1))
        steps_plan = ['residual'] * r + ['topk'] * (prefix_len - r)
    else:
        steps_plan = ['residual'] * prefix_len

    cur_vec = vec
    for step_type in steps_plan:
        if step_type == 'residual':
            tid, score, sims_full = pick_residual_step(cur_vec)
        else:
            tid, score, sims_full = pick_topk_sample(cur_vec)
        chosen.append(tid)
        used.add(tid)
        # residual 更新: cur_vec <- cur_vec - proj_{emb_tid}(cur_vec)
        if mode in ('residual', 'hybrid') and step_type == 'residual':
            basis = emb_norm[tid]
            proj_scale = torch.dot(cur_vec, basis)
            cur_vec = F.normalize(cur_vec - proj_scale * basis, dim=0)
    return chosen

# 基于序列每个位置独立选 token (简单 argmax)

def seq_prefix_from_pred(pred_seq: torch.Tensor, lm: SamOut, allowed_ids=None):
    # pred_seq: (L,D)
    emb = lm.em.weight.data  # (V,D)
    emb_norm = F.normalize(emb, dim=-1)
    if allowed_ids is not None:
        allowed_list = torch.tensor(sorted(list(allowed_ids)), device=emb_norm.device)
    result = []
    for vec in pred_seq:
        v = F.normalize(vec, dim=0)
        sims = torch.matmul(emb_norm, v)  # (V)
        if allowed_ids is not None:
            mask = torch.full_like(sims, -1e9)
            mask[allowed_list] = sims[allowed_list]
            sims = mask
        best = torch.argmax(sims).item()
        result.append(best)
    # 去除可能的 padding 重复或特殊 token
    cleaned = [t for t in result if t not in SPECIAL_TOKEN_IDS]
    # 保留非空前缀最长 LABEL_MAX_TOKENS
    if cleaned:
        return cleaned[:len(result)]
    return result[:4]

# 修改: 生成函数支持多 token 前缀

def generate_from_start_tokens(start_tokens,
                               lm: SamOut,
                               max_length=64,
                               temperature=0.9,
                               top_k=40,
                               top_p=0.9,
                               repetition_penalty=1.2,
                               eos_token=2,
                               vocab: UniVoc = None,
                               debug=False,
                               prefix_text=None,
                               suffix_text=None):
    # 新增: 支持自定义前后缀文本
    prefix_ids = vocab.encode(prefix_text) if (vocab and prefix_text) else []
    suffix_ids = vocab.encode(suffix_text) if (vocab and suffix_text) else []
    input_ids = [1, 5] + prefix_ids + start_tokens + suffix_ids + [6]
    print(f'[Debug] 生成起始 token 序列: {input_ids} -> ' + (vocab.decode(input_ids) if vocab else ''))
    full_sequence = input_ids.copy()
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    state = None
    with torch.no_grad():
        for step in range(max_length):
            logits, state = lm(x, state)
            next_logits = logits[0, -1, :]
            gen_part = full_sequence[len(input_ids):]
            next_id = sample_with_repetition_suppression(next_logits, gen_part, temperature=temperature,
                                                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
            if next_id == eos_token:
                if debug: print(f'[EOS at step {step}]')
                break
            full_sequence.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
            if debug and vocab:
                print(f'step {step}: id={next_id} token={vocab.decode([next_id])}')
    gen_ids = full_sequence[len(input_ids):]
    text = vocab.decode(gen_ids) if vocab else str(gen_ids)
    return text, gen_ids

# 修改: infer_image 使用多 token 前缀

def infer_image(image_path: str,
                max_length: int = 64,
                top_k_embed: int = 5,
                generation_top_k: int = 40,
                top_p: float = 0.9,
                temperature: float = 0.9,
                repetition_penalty: float = 1.2,
                prefix_mode: str = 'hybrid',
                prefix_len: int = 4,
                prefix_source: str = 'mean',  # 'mean' 或 'sequence'
                debug=False,
                prompt_prefix_text: str = None,
                prompt_suffix_text: str = None):
    vocab, lm, vae, mapping, allowed_ids = load_models()
    with torch.no_grad():
        img = load_image(image_path)
        mean, logvar = vae.encode(img)
        z = vae.reparameterize(mean, logvar).squeeze(0)
        pred_seq = mapping(z.unsqueeze(0).to(device)).squeeze(0)  # (L,D)
        if prefix_source == 'sequence':
            start_tokens_multi = seq_prefix_from_pred(pred_seq, lm, allowed_ids=allowed_ids)
            start_tokens_multi = start_tokens_multi[:prefix_len]
            pooled_pred = pred_seq.mean(dim=0)
        else:
            pooled_pred = pred_seq.mean(dim=0)
            start_tokens_multi = multi_token_prefix(pooled_pred, lm, prefix_len=prefix_len, mode=prefix_mode,
                                                    candidate_k=top_k_embed, temperature=0.7, allowed_ids=allowed_ids)
        single_ids, single_scores = embedding_to_start_tokens(pooled_pred, lm, top_k=top_k_embed, allowed_ids=allowed_ids)
    if debug:
        print('[Single top-k candidates](filtered):')
        for tid, sc in zip(single_ids, single_scores):
            print(f'  id={tid} score={sc:.4f} token={vocab.decode([tid])}')
        print(f'[Prefix source={prefix_source} mode={prefix_mode}]:', start_tokens_multi, '->', ''.join(vocab.decode(start_tokens_multi)))
    text, gen_ids = generate_from_start_tokens(start_tokens_multi, lm, max_length=max_length,
                                               temperature=temperature, top_k=generation_top_k, top_p=top_p,
                                               repetition_penalty=repetition_penalty, vocab=vocab, debug=debug,
                                               prefix_text=prompt_prefix_text, suffix_text=prompt_suffix_text)
    return {
        'image_path': image_path,
        'prefix_tokens': start_tokens_multi,
        'prefix_text': vocab.decode(start_tokens_multi),
        'generated_text': text,
        'generation_token_ids': gen_ids
    }

# 与训练一致的序列 MSE 损失

def sequence_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1)
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

# 给定参考文本计算映射损失

def infer_with_text(image_path: str,
                    reference_text: str,
                    debug: bool = True):
    vocab, lm, vae, mapping, _ = load_models()
    tokenizer = vocab
    with torch.no_grad():
        img = load_image(image_path)
        mean, logvar = vae.encode(img)
        z = vae.reparameterize(mean, logvar)
        pred_seq = mapping(z).squeeze(0)  # (L,D)
        # 参考文本 -> token ids
        ref_ids = tokenizer.encode(reference_text)[:pred_seq.size(0)]
        length = len(ref_ids)
        if length < pred_seq.size(0):
            ref_ids = ref_ids + [0]*(pred_seq.size(0)-length)
        token_tensor = torch.tensor([ref_ids], dtype=torch.long, device=device)
        seq_embed = lm.em(token_tensor).squeeze(0)  # (L,D)
        mask = torch.zeros(pred_seq.size(0), device=device)
        mask[:length] = 1.0
        loss = sequence_mse_loss(pred_seq.unsqueeze(0), seq_embed.unsqueeze(0), mask.unsqueeze(0)).item()
        # 最近邻 token 预测
        emb_table = lm.em.weight.data
        emb_norm = F.normalize(emb_table, dim=-1)
        pred_norm = F.normalize(pred_seq[:length], dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.t())  # (len,V)
        nn_ids = sims.argmax(dim=-1).tolist()
        ref_text_trim = tokenizer.decode(ref_ids[:length])
        pred_text = tokenizer.decode(nn_ids)
        if debug:
            print(f'[InferWithText] ref_len={length} loss={loss:.4f}')
            print(f'GT:   {ref_text_trim}')
            print(f'PRED: {pred_text}')
        return {
            'image_path': image_path,
            'reference_text': ref_text_trim,
            'pred_text': pred_text,
            'loss': loss,
            'gt_token_ids': ref_ids[:length],
            'pred_token_ids': nn_ids
        }

# 修改演示: 传入 prefix_mode / prefix_len
if __name__ == '__main__':
    test_image = './tmpC74F.png'
    prompt_prefix_text = '请根据下面的内容写一段话\r\n'
    prompt_postfix_text = '\r\n使用中文'
    if os.path.exists(test_image):
        out = infer_image(test_image, debug=True, max_length=50, prefix_mode='hybrid', 
                          prefix_len=10, prefix_source='sequence',prompt_prefix_text=prompt_prefix_text,
                          prompt_suffix_text=prompt_postfix_text)
        print('\n=== 推理结果 ===')
        for k, v in out.items():
            print(f'{k}: {v}')
        # 参考文本评估示例（需替换成真实类别文本）
        eval_out = infer_with_text(test_image, 'Bear', debug=True)
        print('\n=== 参考文本评估 ===')
        for k, v in eval_out.items():
            print(f'{k}: {v}')
    else:
        print('示例图像不存在, 请修改 test_image 路径。')
