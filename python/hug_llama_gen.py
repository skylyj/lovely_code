from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_dir = './python/llama_random'
model = AutoModelForCausalLM.from_pretrained(model_dir)
# tokenizer = AutoTokenizer.from_pretrained("allenai/llama-small")  # 使用合适的分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# 准备输入
input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")

# 生成文本
output_sequences = model.generate(input_ids, max_length=50)

# 解码生成的文本
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("Generated Text:\n", generated_text)
