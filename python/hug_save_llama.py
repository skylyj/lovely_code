from transformers import LlamaForCausalLM, LlamaConfig
import os

# 指定保存模型的目录
model_dir = './llama_random'

# 确保目录存在
os.makedirs(model_dir, exist_ok=True)

# 初始化LLaMA模型的配置
config = LlamaConfig(
    vocab_size=50265,  # 示例词汇表大小
    hidden_size=768,  # 隐藏层大小
    num_hidden_layers=12,  # Transformer层数
    num_attention_heads=12,  # 注意力头数量
    intermediate_size=3072,  # 前馈网络大小
    max_position_embeddings=1024,  # 最大位置嵌入数
)

# 使用配置初始化模型
model = LlamaForCausalLM(config)

# 保存模型和配置到本地
model.save_pretrained(model_dir)
