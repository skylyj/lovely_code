from transformers import LlamaModel, LlamaConfig
import torch

# 配置模型参数（可以根据需要进行调整）
config = LlamaConfig(
    vocab_size=50265,  # LLaMA 使用的词汇表大小
    hidden_size=128,  # 隐藏层大小
    num_hidden_layers=2,  # Transformer 层数
    num_attention_heads=4,  # 注意力头数量
    intermediate_size=512,  # 前馈网络的大小
    max_position_embeddings=512  # 最大位置嵌入数
)

# 实例化模型
model = LlamaModel(config)

# 生成一些随机数据作为输入
batch_size = 2
seq_length = 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

# 前向传播，获取模型输出
outputs = model(input_ids=input_ids)

# 输出最后一层的隐藏状态的维度
last_hidden_states = outputs.last_hidden_state
print("Last hidden states shape:", last_hidden_states.shape)

# 观察注意力矩阵的维度（如果需要）
# 注意：这需要修改 LLaMA 模型的源代码以返回注意力矩阵，或使用输出中已提供的相关信息
