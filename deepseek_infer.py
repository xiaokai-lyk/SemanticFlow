from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
model.eval()

prompt = "你是一个AI助手，根据用户的提问，给出详细的回答。\r\n什么是sea buckthorn\r\n<think>\n</think>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print("开始生成...")
outputs = model.generate(**inputs, max_new_tokens=2048,
                         pad_token_id=tokenizer.eos_token_id)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

input_ids = inputs["input_ids"]  # shape: [batch_size, seq_len]
embeddings = model.get_input_embeddings()(input_ids)
print("Prompt token ids:", input_ids)
print("Prompt embeddings shape:", embeddings.shape)
print("Prompt embeddings:", embeddings[0])