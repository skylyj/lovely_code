import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device("mps")  # 使用 Metal Performance Shaders (MPS)

# 准备数据
data = [
    ("我爱吃苹果", "我喜欢水果", 1),
    ("今天天气很好", "去公园玩吧", 0),
    ("我正在学习", "我在看书", 1),
    ("猫是一种宠物", "狗是人类的好朋友", 0),
]

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Encode the data
inputs = []
masks = []
labels = []

for text_a, text_b, label in data:
    encoded_dict = tokenizer.encode_plus(
        text_a, text_b,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    inputs.append(encoded_dict['input_ids'])
    masks.append(encoded_dict['attention_mask'])
    labels.append(label)

inputs = torch.cat(inputs, dim=0)
masks = torch.cat(masks, dim=0)
labels = torch.tensor(labels)

# 创建数据集和数据加载器
dataset = TensorDataset(inputs, masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

# 设置优化器和训练模型
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 4
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} training loss: {avg_train_loss:.2f}")

# 评估模型
model.eval()
eval_accuracy = 0
nb_eval_steps = 0

for batch in val_loader:
    b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions = np.argmax(logits, axis=1)
    eval_accuracy += accuracy_score(label_ids, predictions)
    nb_eval_steps += 1

print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps:.2f}")
