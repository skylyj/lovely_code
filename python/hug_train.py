from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size=128):
        self.tokenizer = tokenizer
        self.input_ids = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            if len(tokenized_text) > block_size:
                tokenized_text = tokenized_text[:block_size]
            self.input_ids.append(tokenized_text)

        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

# 生成假的文本数据


def generate_fake_data(num_texts=10, text_length=50):
    texts = []
    for _ in range(num_texts):
        fake_text = " ".join(np.random.choice(["hello", "world", "this", "is", "a", "test"], text_length))
        texts.append(fake_text)
    return texts


texts = generate_fake_data()

# 加载预训练的tokenizer和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 准备数据集
dataset = TextDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 设置训练参数
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cpu")
model.to(device)

# 训练循环
breakpoint()
model.train()
epochs = 3
for epoch in range(epochs):
    for batch in dataloader:
        inputs = batch.to(device)
        breakpoint()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
