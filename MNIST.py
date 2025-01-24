import torch.cuda
from torch import nn    #导入神经网络模块
from torch.utils.data import DataLoader  #数据包管理工具
from torchvision import datasets    #数据处理工具，专门用于图像处理的包
from torchvision.transforms import ToTensor  #数据转换，张量

'''下载训练集数据集(包含训练图片和标签)'''
#datasets.MNIST来加载MNIST数据集作为训练数据集。
#root='data'：指定数据集存储的根目录，可以根据需要进行更改。
#train=True：表示加载训练数据集
#download=True：如果数据集在指定路径中不存在，将自动从官方源下载并保存。
#transform=ToTensor()：指定数据转换操作，将图像数据转换为PyTorch中的Tensor张量格式。
training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),  #张量
)     #对于pythorch库能够识别的数据一般是tensor张量


test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)
# print(len(training_data))

'''展示手写字图片，把训练数据集中前9张图片展示'''
from matplotlib import pyplot as plt
figure = plt.figure()
for i in range(9):
#这一行代码从训练数据集（training_data）中获取第i+10000个数据样本的图像（img）和标签（lable）
    img,lable = training_data[i+10000]
    figure.add_subplot(3,3,i+1)  #这一行代码将一个3x3的子图添加到Figure对象中，将第i+1个子图作为当前子图。
    plt.title(lable)
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap='gray')    #img.squeeze  将维度进行压缩
# plt.show()


'''创建数据DataLoader（数据加载器）
    bath_size:将数据集分成多份，每一份为bath_size个数据
    优点：可以减少内存的使用，提高训练的速度
'''

'''64: 表示批次大小（batch size），即该张量包含了64个样本。
1: 表示通道数（channel），在这里是灰度图像，所以通道数为1。
28: 表示图像的高度（height），即每个图像有28个像素点的高度。
28: 表示图像的宽度（width），即每个图像有28个像素点的宽度。'''

train_dataloader = DataLoader(training_data,batch_size=64)  #64张图片为一个包
test_dataloader = DataLoader(test_data,batch_size=64)
for X,Y in train_dataloader:  #X表示打包好的每一个数据包
    print(f'Shape of X[N,C,H,W]:{X.shape}')
    print(f'Shape of Y:{Y.shape}{Y.dtype}')
    break

'''判断当前设备是否支持GPU,其中mps是苹果m系列芯片的GPU'''
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using  {device}  device')



'''创建神经网络模型'''
class NeuralNetwork(nn.Module):   #通过调用类的形式来使用神经网络
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()     #展开，创建一个展开对象flatten
        self.hidden1 = nn.Linear(28*28,128)
        self.hidden2 = nn.Linear(128,64)
        self.hidden3 = nn.Linear(64,64)
        self.out = nn.Linear(64,10)
    def forward(self,x):
        x = self.flatten(x)   #图像进行展开
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.out(x)
        return x
model = NeuralNetwork().to(device)
print(model)

print('权重个数为：',((28*28)+1)*128+129*256+257*10)


def train(dataloader,model,loss_fn,optimizer):
    model.train()

    batch_size_num = 1
    for x,y in dataloader:
        x,y = x.to(device),y.to(device)
        pred = model.forward(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        print(f'loss:{loss_value:>7f}[num:{batch_size_num}]')
        batch_size_num += 1

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0015)  #可以用Adam 最好的模型
train(train_dataloader,model,loss_fn,optimizer)

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  #测试
    test_loss,correct = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = model.forward(x)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches  #平均正确率
    correct /= size
    print(f'Test result: \n Accuracy:{(100*correct)}%,Avg loss: {test_loss}')

epochs = 1
for t in range(epochs):
    print(f'Epochs {t+1}\n------')
    train(train_dataloader,model,loss_fn, optimizer)
    print('Done!')
test(test_dataloader,model,loss_fn)