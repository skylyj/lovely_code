from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# 检查是否有可用的GPU，如果没有，则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的GPT-2模型和分词器
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

# 生成假数据
text = "Hello, GPT-2. " * 128  # 简单重复一些文本来模拟数据
with open("fake_data.txt", "w") as file:
    file.write(text)

# 准备数据集和数据收集器
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="fake_data.txt",
    block_size=32)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2_toy_output",  # 输出目录
    overwrite_output_dir=True,
    num_train_epochs=1,  # 训练轮次
    per_device_train_batch_size=2,  # 训练批次大小
    save_steps=10_000,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 开始训练
# breakpoint()
trainer.train()
print("Training complete!")
