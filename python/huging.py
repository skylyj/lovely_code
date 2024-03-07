import ipdb
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 编码输入文本，增加终止符
input_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog", return_tensors="pt")

# 生成文本
# ipdb.set_trace()
breakpoint()
output_sequences = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码并打印输出文本
for generated_sequence in output_sequences:
    print(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True))
