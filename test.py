# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# 生成数据
# 生成100个随机点
x = np.random.rand(100, 1)
# 生成对应的y值，y = 3x + 2 + 噪声
y = 3 * x + 2 + np.random.randn(100, 1) / 1.5

# 转换为Tensor
x_tensor = torch.from_numpy(x.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))


# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# 初始化模型、损失函数和优化器
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    # 前向传播
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if epoch % 100 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, 1000, loss.item()))

# 预测
predicted = model(x_tensor).detach().numpy()

# 绘制图像
plt.plot(x, y, "bo", label="Original data")
plt.plot(x, predicted, "r", label="Fitted line")
plt.legend()
plt.show()


print("hello")
print("123")
