import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 定义神经网络模型
class HandwritingClassifier(nn.Module):
    def __init__(self):
        super(HandwritingClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 卷积层 1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 卷积层 2
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 全连接层 1
        self.fc2 = nn.Linear(128, 26)  # 全连接层 2

    def forward(self, x):
        x = self.conv1(x)  # 卷积操作 1
        x = torch.relu(x)  # 激活函数 ReLU
        x = self.pool(x)  # 池化操作
        x = self.conv2(x)  # 卷积操作 2
        x = torch.relu(x)  # 激活函数 ReLU
        x = self.pool(x)  # 池化操作
        x = x.view(-1, 32 * 7 * 7)  # 调整维度
        x = self.fc1(x)  # 全连接层 1
        x = torch.relu(x)  # 激活函数 ReLU
        x = self.fc2(x)  # 全连接层 2
        return x


# 加载数据
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建模型实例
model = HandwritingClassifier()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
