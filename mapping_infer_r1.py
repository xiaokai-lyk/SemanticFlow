import os
import pickle
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Optional

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from MappingModelwithR1 import (
    MappingNetwork,
    LATENT_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT_RATE,
    LABEL_MAX_TOKENS,
    SAVE_DIR,
    VAE_MODEL_PATH,
    DEEPSEEK_MODEL_PATH
)
from BasicVAE.ImageVAE import ImageVAE
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] {device}')

# 默认映射模型路径
DEFAULT_MAPPING_CKPT = os.path.join(SAVE_DIR, 'best_mapping_model_seq.pth')

# 图像预处理
_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    return _transform(img).unsqueeze(0).to(device)

# 加载组件
def load_components(mapping_ckpt: str = DEFAULT_MAPPING_CKPT):
    if not os.path.exists(mapping_ckpt):
        raise FileNotFoundError(f'未找到映射模型权重: {mapping_ckpt}')
    # tokenizer & LM
    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_PATH).to(device)
    lm.eval()
    embed_dim = lm.get_input_embeddings().embedding_dim
    # VAE
    vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_MODEL_PATH, map_location=device))
    vae.eval()
    # Mapping
    mapping = MappingNetwork(
        input_dim=LATENT_DIM,
        output_dim=embed_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE,
        seq_len=LABEL_MAX_TOKENS
    ).to(device)
    mapping.load_state_dict(torch.load(mapping_ckpt, map_location=device))
    mapping.eval()
    # 可选标签 token 集
    label_token_set_path = os.path.join(SAVE_DIR, 'label_token_set.pkl')
    allowed_ids = None
    if os.path.exists(label_token_set_path):
        with open(label_token_set_path, 'rb') as f:
            allowed_ids = set(pickle.load(f))
        print(f'[Info] 载入标签 token 集合, 大小: {len(allowed_ids)}')
    else:
        print('[Warn] 未找到标签 token 集合, 不限制候选。')
    return tokenizer, lm, vae, mapping, allowed_ids

# ========== Token 选择工具 ==========
SPECIAL_TOKEN_IDS = set()  # HuggingFace 模型一般自行处理, 这里仅用于前缀过滤

def embedding_to_start_tokens(mean_embed: torch.Tensor, lm, top_k: int = 5, allowed_ids=None):
    emb = lm.get_input_embeddings().weight.data  # (V,D)
    pred = F.normalize(mean_embed.unsqueeze(0), dim=-1)          # (1,D)
    emb_norm = F.normalize(emb, dim=-1)                          # (V,D)
    sims = torch.matmul(pred, emb_norm.t()).squeeze(0)           # (V)
    if allowed_ids is not None:
        mask = torch.full_like(sims, -1e9)
        idx = torch.tensor(list(allowed_ids), dtype=torch.long, device=sims.device)
        mask[idx] = sims[idx]
        sims = mask
    k = min(top_k, (sims > -1e9/2).sum().item())
    topv, topi = torch.topk(sims, k=k)
    return topi.tolist(), topv.tolist()


def seq_prefix_from_pred(pred_seq: torch.Tensor, lm, allowed_ids=None):
    # pred_seq: (L,D)
    emb = lm.get_input_embeddings().weight.data
    emb_norm = F.normalize(emb, dim=-1)
    res = []
    allowed_tensor = None
    if allowed_ids is not None:
        allowed_tensor = torch.tensor(sorted(list(allowed_ids)), device=pred_seq.device)
    for vec in pred_seq:
        v = F.normalize(vec, dim=0)
        sims = torch.matmul(emb_norm, v)  # (V)
        if allowed_tensor is not None:
            mask = torch.full_like(sims, -1e9)
            mask[allowed_tensor] = sims[allowed_tensor]
            sims = mask
        best = torch.argmax(sims).item()
        if best not in SPECIAL_TOKEN_IDS:
            res.append(best)
    return res


def multi_token_prefix(mean_embed: torch.Tensor,
                       lm,
                       prefix_len: int = 4,
                       mode: str = 'hybrid',
                       candidate_k: int = 40,
                       temperature: float = 0.7,
                       allowed_ids=None,
                       allow_repetition: bool = False):
    emb = lm.get_input_embeddings().weight.data
    emb_norm = F.normalize(emb, dim=-1)
    vec = F.normalize(mean_embed, dim=0)
    V = emb_norm.size(0)
    allowed_tensor = None
    if allowed_ids is not None:
        allowed_tensor = torch.tensor(sorted(list(allowed_ids)), device=emb_norm.device)
    chosen, used = [], set()

    def mask_sims(sims):
        if allowed_tensor is not None:
            temp = torch.full_like(sims, -1e9)
            temp[allowed_tensor] = sims[allowed_tensor]
            sims = temp
        if not allow_repetition:
            for tid in used:
                sims[tid] = -1e9
        for tid in SPECIAL_TOKEN_IDS:
            if 0 <= tid < V:
                sims[tid] = -1e9
        return sims

    def step_residual(cur):
        sims = torch.matmul(cur, emb_norm.t())
        sims_m = mask_sims(sims.clone())
        topv, topi = torch.topk(sims_m, k=1)
        return topi.item(), topv.item()

    def step_topk(cur):
        sims = torch.matmul(cur, emb_norm.t())
        sims_m = mask_sims(sims.clone())
        valid = (sims_m > -1e9/2).sum().item()
        k = min(candidate_k, valid if valid > 0 else 1)
        topv, topi = torch.topk(sims_m, k=k)
        logits = topv / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        sel = torch.multinomial(probs, 1).item()
        return topi[sel].item(), probs[sel].item()

    if mode == 'residual':
        plan = ['residual'] * prefix_len
    elif mode == 'topk_sample':
        plan = ['topk'] * prefix_len
    elif mode == 'hybrid':
        r = max(1, min(2, prefix_len))
        plan = ['residual'] * r + ['topk'] * (prefix_len - r)
    else:
        plan = ['residual'] * prefix_len

    cur_vec = vec
    for t in plan:
        if t == 'residual':
            tid, _ = step_residual(cur_vec)
        else:
            tid, _ = step_topk(cur_vec)
        chosen.append(tid)
        used.add(tid)
        if t == 'residual' and mode in ('residual','hybrid'):
            basis = emb_norm[tid]
            proj = torch.dot(cur_vec, basis)
            cur_vec = F.normalize(cur_vec - proj * basis, dim=0)
    return chosen

# ========== 生成 ==========

def build_input_ids(tokenizer, prefix_text_ids: List[int], mapped_tokens: List[int], suffix_text_ids: List[int]):
    ids = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
    ids.extend(prefix_text_ids)
    ids.extend(mapped_tokens)
    ids.extend(suffix_text_ids)
    return ids


def generate_text(tokenizer, lm, input_ids: List[int] = None, max_new_tokens=64, temperature=0.9, top_k=40, top_p=0.9, repetition_penalty=1.2, do_sample=True, inputs_embeds: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None):
    """支持两种输入: 传统 input_ids 或 直接的 inputs_embeds (pred_seq 等)。
    若使用 inputs_embeds, 生成结果只包含新增 token 序列。"""
    gen_kwargs = dict(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    with torch.no_grad():
        if inputs_embeds is not None:
            # inputs_embeds: (1, L, D)
            if attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
            out = lm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)
            new_tokens = out[0].tolist()  # 全部为新 token
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text, new_tokens
        else:
            inp = torch.tensor([input_ids], dtype=torch.long, device=device)
            out = lm.generate(inp, **gen_kwargs)
            gen = out[0].tolist()
            new_tokens = gen[len(input_ids):]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text, new_tokens

# ========== 主推理函数 ==========

def infer_image(
    image_path: str,
    mapping_ckpt: str = DEFAULT_MAPPING_CKPT,
    prefix_mode: str = 'hybrid',
    prefix_len: int = 4,
    prefix_source: str = 'sequence',  # 'sequence' or 'mean'
    top_k_embed: int = 5,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    prompt_prefix_text: Optional[str] = None,
    prompt_suffix_text: Optional[str] = None,
    debug: bool = False,
    do_sample: bool = True,
    use_raw_pred_seq: bool = False
) -> Dict[str, Any]:
    tokenizer, lm, vae, mapping, allowed_ids = load_components(mapping_ckpt)
    with torch.no_grad():
        img = load_image(image_path)
        mean, logvar = vae.encode(img)
        z = vae.reparameterize(mean, logvar)  # (1, latent)
        pred_seq = mapping(z).squeeze(0)      # (L, D)

        if not use_raw_pred_seq:
            if prefix_source == 'sequence':
                start_tokens = seq_prefix_from_pred(pred_seq, lm, allowed_ids=allowed_ids)[:prefix_len]
                pooled = pred_seq.mean(dim=0)
            else:
                pooled = pred_seq.mean(dim=0)
                start_tokens = multi_token_prefix(pooled, lm, prefix_len=prefix_len, mode=prefix_mode, candidate_k=top_k_embed, temperature=0.7, allowed_ids=allowed_ids)
            single_ids, single_scores = embedding_to_start_tokens(pooled, lm, top_k=top_k_embed, allowed_ids=allowed_ids)
        else:
            start_tokens = []  # 不再需要离散 token
            single_ids, single_scores = [], []

    prefix_ids = tokenizer.encode(prompt_prefix_text, add_special_tokens=False) if prompt_prefix_text else []
    suffix_ids = tokenizer.encode(prompt_suffix_text, add_special_tokens=False) if prompt_suffix_text else []

    if debug and not use_raw_pred_seq:
        print('[Single top-k candidates]:')
        for tid, sc in zip(single_ids, single_scores):
            print(f'  id={tid} score={sc:.4f} token={tokenizer.decode([tid], skip_special_tokens=True)}')
        print(f'[Prefix] mode={prefix_mode} source={prefix_source}:', start_tokens, '->', tokenizer.decode(start_tokens, skip_special_tokens=True))

    if use_raw_pred_seq:
        # 构造 inputs_embeds: [BOS] + prefix_text + pred_seq + suffix_text
        embed_layer = lm.get_input_embeddings()
        pieces = []
        if tokenizer.bos_token_id is not None:
            bos_emb = embed_layer.weight[tokenizer.bos_token_id].unsqueeze(0)
            pieces.append(bos_emb)
        if prefix_ids:
            prefix_emb = embed_layer(torch.tensor(prefix_ids, device=device))
            pieces.append(prefix_emb)
        pieces.append(pred_seq)  # 直接放入映射序列
        if suffix_ids:
            suffix_emb = embed_layer(torch.tensor(suffix_ids, device=device))
            pieces.append(suffix_emb)
        full_emb = torch.cat(pieces, dim=0).unsqueeze(0)  # (1, L, D)
        if debug:
            print('[Debug] 使用 raw pred_seq 作为前缀嵌入, total_len=', full_emb.size(1))
        gen_text, gen_token_ids = generate_text(
            tokenizer, lm, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, do_sample=do_sample, inputs_embeds=full_emb
        )
    else:
        input_ids = build_input_ids(tokenizer, prefix_ids, start_tokens, suffix_ids)
        if debug:
            print('[Debug] 输入 tokens:', input_ids)
            print('[Debug] 输入解码(含映射前缀):', tokenizer.decode(input_ids, skip_special_tokens=True))
        gen_text, gen_token_ids = generate_text(
            tokenizer, lm, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )

    return {
        'image_path': image_path,
        'prefix_tokens': start_tokens,
        'prefix_text': tokenizer.decode(start_tokens, skip_special_tokens=True) if start_tokens else '',
        'generated_text': gen_text,
        'generated_token_ids': gen_token_ids,
        'used_raw_pred_seq': use_raw_pred_seq
    }

# ========== 文本参考评估（可选） ==========

def sequence_mse_loss(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)

def infer_with_text(image_path: str, reference_text: str, mapping_ckpt: str = DEFAULT_MAPPING_CKPT, debug: bool = True, prompt_prefix_text: Optional[str] = None, prompt_suffix_text: Optional[str] = None):
    tokenizer, lm, vae, mapping, _ = load_components(mapping_ckpt)
    with torch.no_grad():
        img = load_image(image_path)
        mean, logvar = vae.encode(img)
        z = vae.reparameterize(mean, logvar)
        pred_seq = mapping(z).squeeze(0)  # (L,D)
        ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)[:LABEL_MAX_TOKENS]
        length = len(ref_ids)
        if length < LABEL_MAX_TOKENS:
            ref_ids = ref_ids + [tokenizer.pad_token_id]*(LABEL_MAX_TOKENS - length)
        ref_tensor = torch.tensor([ref_ids], dtype=torch.long, device=device)
        seq_embed = lm.get_input_embeddings()(ref_tensor).squeeze(0)
        mask = torch.zeros(LABEL_MAX_TOKENS, device=device)
        mask[:length] = 1.0
        loss = sequence_mse_loss(pred_seq, seq_embed, mask).item()
        emb_tab = lm.get_input_embeddings().weight.data
        emb_norm = F.normalize(emb_tab, dim=-1)
        pred_norm = F.normalize(pred_seq[:length], dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.t())
        nn_ids = sims.argmax(dim=-1).tolist()
        ref_text_trim = tokenizer.decode(ref_ids[:length], skip_special_tokens=True)
        pred_text = tokenizer.decode(nn_ids, skip_special_tokens=True)
        if debug:
            print(f'[InferWithText] ref_len={length} loss={loss:.4f}')
            print('GT:', ref_text_trim)
            print('PRED:', pred_text)
        return {
            'image_path': image_path,
            'reference_text': ref_text_trim,
            'pred_text': pred_text,
            'loss': loss,
            'gt_token_ids': ref_ids[:length],
            'pred_token_ids': nn_ids
        }

# ========== WebUI 会话封装类 ==========
class MappingChatSession:
    """封装图片->文本映射与多轮对话生成逻辑, 便于 WebUI 复用。"""
    def __init__(self, mapping_ckpt: str = DEFAULT_MAPPING_CKPT, prefix_mode: str = 'hybrid', prefix_len: int = 4, prefix_source: str = 'sequence', top_k_embed: int = 5):
        self.mapping_ckpt = mapping_ckpt
        self.tokenizer, self.lm, self.vae, self.mapping, self.allowed_ids = load_components(mapping_ckpt)
        self.prefix_mode = prefix_mode
        self.prefix_len = prefix_len
        self.prefix_source = prefix_source
        self.top_k_embed = top_k_embed
        self.image_path: Optional[str] = None
        self.pred_seq: Optional[torch.Tensor] = None  # (L,D)
        self.z: Optional[torch.Tensor] = None
        self.system_prompt: str = ''
        self.history: List[Dict[str,str]] = []  # [{'role':'user'/'assistant','content':...}]

    # --------------- 基础 ---------------
    def set_system_prompt(self, text: str):
        self.system_prompt = text or ''

    def reset_conversation(self):
        self.history.clear()

    def get_history(self) -> List[Dict[str,str]]:
        return list(self.history)

    def set_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'图像不存在: {image_path}')
        self.image_path = image_path
        with torch.no_grad():
            img = load_image(image_path)
            mean, logvar = self.vae.encode(img)
            self.z = self.vae.reparameterize(mean, logvar)
            self.pred_seq = self.mapping(self.z).squeeze(0)  # (L,D)

    # --------------- 内部辅助 ---------------
    def _build_prefix_tokens(self, pred_seq: torch.Tensor):
        if self.prefix_source == 'sequence':
            start_tokens = seq_prefix_from_pred(pred_seq, self.lm, allowed_ids=self.allowed_ids)[:self.prefix_len]
            pooled = pred_seq.mean(dim=0)
        else:
            pooled = pred_seq.mean(dim=0)
            start_tokens = multi_token_prefix(pooled, self.lm, prefix_len=self.prefix_len, mode=self.prefix_mode, candidate_k=self.top_k_embed, temperature=0.7, allowed_ids=self.allowed_ids)
        return start_tokens, pooled

    def _generate(self, convo_prefix: str, user_input: str, suffix_text: Optional[str], max_new_tokens=128, temperature=0.9, top_k=40, top_p=0.9, repetition_penalty=1.2, do_sample=True, use_raw_pred_seq=False):
        if self.pred_seq is None:
            raise RuntimeError('尚未设置图片，请先调用 set_image().')
        tokenizer = self.tokenizer
        lm = self.lm
        pred_seq = self.pred_seq
        prefix_ids = tokenizer.encode(convo_prefix + user_input, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False) if suffix_text else []
        if use_raw_pred_seq:
            embed_layer = lm.get_input_embeddings()
            pieces = []
            if tokenizer.bos_token_id is not None:
                pieces.append(embed_layer.weight[tokenizer.bos_token_id].unsqueeze(0))
            if prefix_ids:
                pieces.append(embed_layer(torch.tensor(prefix_ids, device=device)))
            pieces.append(pred_seq)
            if suffix_ids:
                pieces.append(embed_layer(torch.tensor(suffix_ids, device=device)))
            full_emb = torch.cat(pieces, dim=0).unsqueeze(0)
            text, _ = generate_text(tokenizer, lm, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample, inputs_embeds=full_emb)
            return text
        else:
            start_tokens, _ = self._build_prefix_tokens(pred_seq)
            input_ids = build_input_ids(tokenizer, prefix_ids, start_tokens, suffix_ids)
            text, _ = generate_text(tokenizer, lm, input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, do_sample=do_sample)
            return text

    # --------------- 对话接口 ---------------
    def chat(self, user_input: str, suffix_text: Optional[str] = None, **gen_kwargs) -> str:
        convo_prefix_parts = []
        if self.system_prompt:
            convo_prefix_parts.append(f"[System]\n{self.system_prompt}\n")
        for turn in self.history:
            role = 'User' if turn['role']=='user' else 'Assistant'
            convo_prefix_parts.append(f"[{role}] {turn['content']}\n")
        convo_prefix = ''.join(convo_prefix_parts) + '[User] '
        assistant_text = self._generate(convo_prefix, user_input + '\n[Assistant] ', suffix_text, **gen_kwargs)
        self.history.append({'role':'user','content':user_input})
        self.history.append({'role':'assistant','content':assistant_text})
        return assistant_text

    # --------------- 参考文本评估 ---------------
    def evaluate_with_reference(self, reference_text: str) -> Dict[str, Any]:
        if self.pred_seq is None:
            raise RuntimeError('尚未设置图片，请先调用 set_image().')
        tokenizer = self.tokenizer
        lm = self.lm
        pred_seq = self.pred_seq  # (L,D)
        ref_ids = tokenizer.encode(reference_text, add_special_tokens=False)[:LABEL_MAX_TOKENS]
        length = len(ref_ids)
        if length < LABEL_MAX_TOKENS:
            ref_ids = ref_ids + [tokenizer.pad_token_id]*(LABEL_MAX_TOKENS - length)
        ref_tensor = torch.tensor([ref_ids], dtype=torch.long, device=device)
        seq_embed = lm.get_input_embeddings()(ref_tensor).squeeze(0)
        mask = torch.zeros(LABEL_MAX_TOKENS, device=device)
        mask[:length] = 1.0
        loss = sequence_mse_loss(pred_seq, seq_embed, mask).item()
        emb_tab = lm.get_input_embeddings().weight.data
        emb_norm = F.normalize(emb_tab, dim=-1)
        pred_norm = F.normalize(pred_seq[:length], dim=-1)
        sims = torch.matmul(pred_norm, emb_norm.t())
        nn_ids = sims.argmax(dim=-1).tolist()
        ref_text_trim = tokenizer.decode(ref_ids[:length], skip_special_tokens=True)
        pred_text = tokenizer.decode(nn_ids, skip_special_tokens=True)
        return {
            'image_path': self.image_path,
            'reference_text': ref_text_trim,
            'pred_text': pred_text,
            'loss': loss,
            'gt_token_ids': ref_ids[:length],
            'pred_token_ids': nn_ids
        }

# ========== CLI ==========

def parse_args():
    ap = argparse.ArgumentParser(description='DeepSeek-R1 映射推理脚本')
    ap.add_argument('--image', type=str, required=True, help='输入图像路径')
    ap.add_argument('--mapping_ckpt', type=str, default=DEFAULT_MAPPING_CKPT)
    ap.add_argument('--prefix_mode', type=str, default='hybrid')
    ap.add_argument('--prefix_len', type=int, default=4)
    ap.add_argument('--prefix_source', type=str, default='sequence', choices=['sequence','mean'])
    ap.add_argument('--top_k_embed', type=int, default=5)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=40)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.2)
    ap.add_argument('--prompt_prefix_text', type=str, default=None)
    ap.add_argument('--prompt_suffix_text', type=str, default=None)
    ap.add_argument('--no_debug', action='store_false', default=False, help='关闭调试信息')
    ap.add_argument('--ref_text', type=str, default=None, help='可选: 参考文本评估')
    ap.add_argument('--do_sample', action='store_true', default=False, help='是否使用采样生成 (默认False)')
    ap.add_argument('--use_raw_pred_seq', action='store_true', default=False, help='直接将 pred_seq 作为嵌入前缀输入模型')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(f'图像不存在: {args.image}')
    out = infer_image(
        image_path=args.image,
        mapping_ckpt=args.mapping_ckpt,
        prefix_mode=args.prefix_mode,
        prefix_len=args.prefix_len,
        prefix_source=args.prefix_source,
        top_k_embed=args.top_k_embed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        prompt_prefix_text=args.prompt_prefix_text,
        prompt_suffix_text=args.prompt_suffix_text,
        debug=not args.no_debug,
        do_sample=args.do_sample,
        use_raw_pred_seq=args.use_raw_pred_seq
    )
    print('\n=== 推理结果 ===')
    for k,v in out.items():
        print(f'{k}: {v}')
    if args.ref_text:
        eval_out = infer_with_text(args.image, args.ref_text, mapping_ckpt=args.mapping_ckpt, debug=True, prompt_prefix_text=args.prompt_prefix_text, prompt_suffix_text=args.prompt_suffix_text)
        print('\n=== 参考文本评估 ===')
        for k,v in eval_out.items():
            print(f'{k}: {v}')
