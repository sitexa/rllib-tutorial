import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

# 1. 数据预处理
# 读取Excel文件
df = pd.read_excel('SSQ.xlsx')

# 选择所需的列
selected_columns = ['期号', '红一', '红二', '红三', '红四', '红五', '红六', '蓝球']
df = df[selected_columns]

# 将'期号'转换为相对值
df['期号'] = df['期号'] - df['期号'].min()

# 对彩票号码进行归一化
scaler = MinMaxScaler()
lottery_numbers = df[['红一', '红二', '红三', '红四', '红五', '红六', '蓝球']].values
scaled_numbers = scaler.fit_transform(lottery_numbers)

# 2. 准备数据集
class LotteryDataset(Dataset):
    def __init__(self, data, period, sequence_length, noise_level=0.01):
        self.data = torch.FloatTensor(data)
        self.period = torch.LongTensor(period)
        self.sequence_length = sequence_length
        self.noise_level = noise_level

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx+self.sequence_length]
        target = self.data[idx+self.sequence_length]
        
        # 添加随机噪声
        noise = torch.randn_like(sequence) * self.noise_level
        sequence = sequence + noise
        
        return sequence, self.period[idx+self.sequence_length], target

sequence_length = 1600

# 使用带噪声的数据集
dataset = LotteryDataset(scaled_numbers, df['期号'].values, sequence_length, noise_level=0.01)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 定义神经网络模型
class LotteryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LotteryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def custom_lottery_loss(predictions, targets):
    # 确保预测值和目标值都是浮点型
    predictions = predictions.float()
    targets = targets.float()
    
    # 计算每个球的损失
    loss = 0
    for i in range(7):  # 6个红球和1个蓝球
        pred = predictions[:, i]
        target = targets[:, i]
        
        # 完全正确预测的奖励
        correct_prediction = (pred.round() == target).float()
        loss -= correct_prediction.mean()
        
        # 根据预测与实际的接近程度惩罚
        difference = torch.abs(pred - target) / 33  # 假设号码范围是1-33
        loss += difference.mean()
    
    return loss

# 在训练循环中使用这个损失函数
criterion = custom_lottery_loss

# 4. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LotteryPredictor(input_size=7, hidden_size=128, num_layers=3, output_size=7).to(device)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # 添加这行来定义优化器

num_epochs = 1500
best_loss = float('inf')
patience = 20
no_improve = 0

for epoch in range(num_epochs):
    model.train()
    for batch_x, _, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, _, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item()
    
    val_loss /= len(test_loader)
    
    # 学习率调度
    # scheduler.step(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
    
    if no_improve == patience:
        print("Early stopping")
        break
    
# 5. 评估模型
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_x, _, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        test_loss += criterion(outputs, batch_y).item()
    
print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# 6. 预测下一期彩票号码
last_sequence = torch.FloatTensor(scaled_numbers[-sequence_length:]).unsqueeze(0).to(device)
prediction = model(last_sequence)
predicted_numbers = scaler.inverse_transform(prediction.cpu().detach().numpy())

print("预测的下一期彩票号码:")
print(f"期号: {df['期号'].max() + 1}")
print(f"红球: {predicted_numbers[0][:6].round().astype(int)}")
print(f"蓝球: {predicted_numbers[0][6].round().astype(int)}")