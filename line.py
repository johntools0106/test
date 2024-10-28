import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 特征（自变量）
y = 4 + 3 * X + np.random.randn(100, 1)  # 目标变量（因变量）

# 转换为 PyTorch 张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# 创建线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出都是一个特征

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)  # 计算损失
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新权重

    if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 进行预测
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predicted = model(X_tensor).numpy()

# 可视化结果
plt.scatter(X, y, color='blue', label='Actual data')  # 实际数据点
plt.plot(X, predicted, color='red', label='Predicted line')  # 预测的线
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression with PyTorch')
plt.legend()
plt.show()

# 输出模型参数
print("Intercept:", model.linear.bias.item())
print("Slope:", model.linear.weight.item())

