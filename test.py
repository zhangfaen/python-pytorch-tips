import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 假设我们有一个简单的Transformer模型作为策略模型
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Transformer(hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        import pdb
        pdb.set_trace()
        output = self.transformer(src_emb, tgt_emb)
        output = self.fc(output)
        return output

# 自定义数据集，用于偏好优化
class PreferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 奖励计算
def compute_reward(log_probs, length):
    return log_probs.sum(dim=1) / length

# SimPO目标函数
def simpo_loss(policy_model, data, beta=1.0, gamma=0.5):
    src, tgt_winning, tgt_losing = data
    log_probs_winning = policy_model(src, tgt_winning).log_softmax(dim=-1)
    log_probs_losing = policy_model(src, tgt_losing).log_softmax(dim=-1)
    
    length_winning = tgt_winning.size(1)
    length_losing = tgt_losing.size(1)
    
    reward_winning = compute_reward(log_probs_winning, length_winning)
    reward_losing = compute_reward(log_probs_losing, length_losing)
    
    margin = reward_winning - reward_losing - gamma
    loss = -F.logsigmoid(margin * beta).mean()
    
    return loss

# 假设我们有一些数据
data = [
    (torch.randint(0, 100, (10,)), torch.randint(0, 100, (15,)), torch.randint(0, 100, (15,))),
    # 更多数据...
]

# 数据加载器
dataset = PreferenceDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 超参数
vocab_size = 100
hidden_size = 256
num_layers = 2
learning_rate = 0.001

# 模型和优化器
model = SimpleTransformer(vocab_size, hidden_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = simpo_loss(model, batch)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
