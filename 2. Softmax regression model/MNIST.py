import torch
import torchvision  # pytorch 实现计算机视觉用到的一个库
from torch.utils import data
from torchvision import transforms  # 对数据进行操作
import matplotlib.pyplot as plt

# 通过框架中内置的函数将 Fashion-MNIST 数据集下载并读到内存中
# 通过 ToTensor 实例将图像数据从 PIL 类型变换成 32 位浮点数格式
# 并除以 255 使得所有像素的数值均在 0 和 1 之间


trans = transforms.ToTensor()  # 将图片转成张量，即对图片进行预处理
# 将图片下载到上级目录中的 data 中，指定为训练数据集，将其转成张量。
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

# 显示第 0 行第 0 列的张量形状
print(len(mnist_train), len(mnist_test), mnist_train[0][0].shape)


def get_fashion_mnist_labels(labels):
    # 返回 Fashion-MNIST 数据集的文本标签
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(input_img, num_rows, num_cols, titles, scale=1.5):
    # 打印一组图片
    fig_size = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, input_img)):
        if torch.is_tensor(input_img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL 图片
            ax.imshow(img)
    ax.set_title(titles[i])


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y), scale=1.5)

plt.show()

# 读取一小批量数据，大小为 batch_size

batch_size = 256


def get_dataloader_workers():
    # 使用四个进程来读取数据
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())


# 制定训练集，批量大小，随机，进程通过 get_dataloader_workers 函数来获取


# 最后定义函数

def load_data_fashion_mnist(batch_size, resize=None):
    # 下载 Fashion-MNIST 数据集，并加载到内存中
    trans = [transforms.ToTensor()]
    # 之后模型如果更大，可能需要 resize
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return data.DataLoader(mnist_train, batch_size, shuffle=True,
                           num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=True,
                                                                                  num_workers=get_dataloader_workers())


# softmax 回归的从零开始实现

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 接下来「展平所有图片，将它们视为长度为 784 的向量。因为我们的数据集有 10 个类别，所以网络输出维度为 10
num_inputs = 784
num_outputs = 10

# 类似于线性回归的知识，我们需要对每个像素点定义其权重，这里我们选择将其初始化为一个高斯随机分布，形状的行由输入长度决定，列由输出决定。
# 而偏移量由 b 表示，用零来初始化

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 给定一个矩阵，我们可以对所有元素求和

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)


# 接下来要定义 Softmax 操作


def softmax(X):
    X_exp = torch.exp(X)  # 对输入 X 逐元素取指数
    partition = X_exp.sum(1, keepdim=True)  # Softmax 回归中的分母，求和
    return X_exp / partition  # 应用了广播机制


# 接下来实现 Softmax 回归模型

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
    # 在之前我们已经使得 W.shape[0] 为 784，因此这里相当于将 X 变形为长任意，宽为 784 的张量，并采用类似线性回归的方法，最后返回 Softmax 值

# 接下来实现交叉熵的功能。创建一个数据 y_hat ，其中包含 2 个样本在 3 个类别的预测概率，使用 y 作为 y_hat 中概率的索引


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6]], [0.3, 0.2, 0.5])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])
    # 首先，len(y_hat) 访问的是 y_hat 第零维度的长度，相当于样本数
    # 然后创建一个一维张量，内容为 [0,1,2,...,len(y_hat)]
    # 而 [range(len(y_hat)), y] 涉及到 Python 的高级索引语法，意思是以 [0,y_1], [1,y_2], [2,y_3] 的方式取出 y_hat 元素
    # 在这里，它表示的是对于每个样本，只取出其正确标号的 y_hat，也就是预测概率
    # 最后按照交叉熵的定义，取对数，得到对应的损失函数，这是一个长为 len(y_hat) 的一维张量


# 然后将预测类别与真实的 y 元素进行比较

def accuracy(y_hat, y):
    # 计算预测正确的数量
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 若 y_hat 是一个二维矩阵，且至少每个样本有两种类别的可能性
        y_hat = y_hat.argmax(axis=1)  # 则压缩 y_hat 的第 1 维，取原来一行中最大的那个元素的下下标来取代整个行
    # 为了防止 y.dtype 的数据类型和 y 不一样，我们先将 y_hat 转为 y 的数据类型，然后进行比较，由此得到一个由 True 和 False 组成的一维张量
    cmp = y_hat.type(y.dtype) == y
    # 再把 cmp 转成跟 y 一样的类型，此时就应该是数字了，然后求和并返回浮点数，就可以知道预测正确的数量有多少了
    return float(cmp.type(y.dtype).sum())


def evaluate_accuaray(net, data_iter):
    # 纯看不懂，照抄
    # 计算在指定数据集上模型的精度。
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric [1]


class Accumulator:
    # 在 n 个变量上累加
    # 纯看不懂，照抄
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 进行 Softmax 回归训练

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):  # 若使用 nn.Module 实现的话，使用训练模式，告诉 Pytorch 需要计算梯度
        net.train()
    metric = Accumulator(3)  # 用长度为 3 的迭代器获取需要的信息
    for X, y in train_iter:  # 扫一遍数据
        y_hat = net(X)  # 用模型计算 y_hat
        l = loss(y_hat, y)  # 计算交叉熵
        if isinstance(updater, torch.optim.Optimizer):  # 若 updater 是 optim.Optimizer 类型
            updater.zero_grad()  # 先将梯度设置为 0
            l.backward()  # 反向求导计算梯度
            updater.step()  # 对参数自更新
            metric.add(  # 放进累加器
                float(1) * len(y), accuracy(y_hat, y),
                y.size().numel())
        else:  # 如果是自己实现的话
            l.sum().backward()  # l 是个向量，求和再算梯度
            updater(X.shape[0])  # 根据批量大小 update
            metric.add(float(l.sum()),accuracy(y_hat,y), y.numel())  # 记录分类正确个数
    return  metric[0] / metric[2], metric[1] / metric[2]  # 返回损失/样本总数，分类正确/样本总数z