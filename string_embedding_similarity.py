import torch
import torch.nn.functional as F
from SamOutVXP.model3 import SamOut
from SamOutVXP.high_vocab3 import UniVoc

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 路径参数
LM_MODEL_PATH = './SamOutVXP/model_pretrain_cap_1_sft_new.pth'
''
VOC_SIZE = UniVoc().voc_size + 1
TEXT_EMBED_DIM = 512
LABEL_MAX_TOKENS = 8

def get_seq_embedding_and_mask(text, lm_model, tokenizer, device='cpu'):
    token_ids = tokenizer.encode(text)
    token_ids = token_ids[:LABEL_MAX_TOKENS]
    length = len(token_ids)
    if length < LABEL_MAX_TOKENS:
        token_ids = token_ids + [0] * (LABEL_MAX_TOKENS - length)
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        seq_embed = lm_model.em(token_tensor).squeeze(0)  # (L, D)
    mask = torch.zeros(LABEL_MAX_TOKENS, dtype=torch.float32)
    mask[:length] = 1.0
    return seq_embed.cpu(), mask.cpu(), token_ids

def sequence_cosine_loss(pred, target, mask):
    # pred/target: (B,L,D)
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    cos = (pred_n * targ_n).sum(dim=-1)  # (B,L)
    loss = (1.0 - cos) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def vector_norms(seq_embed, mask):
    norms = seq_embed.norm(dim=-1)  # (L,)
    valid_norms = norms[mask.bool()]
    return valid_norms.tolist()

def compare_strings_seq(str1, str2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = SamOut(voc_size=VOC_SIZE, hidden_size=TEXT_EMBED_DIM, num_heads=8, num_layers=8).to(device)
    lm_model.load_state_dict(torch.load(LM_MODEL_PATH, map_location=device))
    lm_model.eval()
    tokenizer = UniVoc()
    seq1, mask1, ids1 = get_seq_embedding_and_mask(str1, lm_model, tokenizer, device)
    seq2, mask2, ids2 = get_seq_embedding_and_mask(str2, lm_model, tokenizer, device)
    # 只对有效 token 计算掩码余弦相似度
    cos_sim = 1.0 - sequence_cosine_loss(seq1, seq2, mask1 * mask2)
    norms1 = vector_norms(seq1, mask1)
    norms2 = vector_norms(seq2, mask2)
    print(f"字符串1: {str1}\n字符串2: {str2}")
    print(f"掩码余弦相似度: {cos_sim:.4f}")
    print(f"字符串1每个token模长: {[f'{n:.4f}' for n in norms1]}")
    print(f"字符串2每个token模长: {[f'{n:.4f}' for n in norms2]}")

if __name__ == "__main__":
    str1 = "苹果"
    str2 = "apple"
    compare_strings_seq(str1, str2)
