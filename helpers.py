from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class BertTextSimilarity:
    def __init__(self, model_path="./bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def calculate(self, text1, text2):
        inputs1 = self.tokenizer(text1, return_tensors='pt', truncation=True, padding=True)
        inputs2 = self.tokenizer(text2, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
        emb1 = outputs1.last_hidden_state[:, 0, :]
        emb2 = outputs2.last_hidden_state[:, 0, :]
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity

# 示例
if __name__ == "__main__":
    text_a = "A dog is barking."
    text_b = "A dog makes noise."
    bert_sim = BertTextSimilarity()
    sim = bert_sim.calculate(text_a, text_b)
    print(f"Similarity: {sim}")