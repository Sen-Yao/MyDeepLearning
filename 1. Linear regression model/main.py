import random  # 用来之后打乱数据
import torch


# 为了执行之后的操作，这里设计一个人造模型来生成具有线性性的数据。
# 根据带有噪声的线性模型构造一个人造数据集
# 我们使用线性模型参数列向量 w，标量 b 和噪声 epsilon 生成数据集及其标签
# w 表示人造模型的参数列向量，b 表示人造模型的偏移量列向量，num_examples 表示需要生成的样本数量
def synthetic_data(synthetic_w, synthetic_b, num_examples):
    # 生成 y = Xw + b + 噪声
    synthetic_x = torch.normal(0, 1, (num_examples, len(synthetic_w)))
    # 生成服从正态分布的张量，其形状需要对应，即 num_examples 个长度为 len(w) 的列向量构成的张量
    synthetic_y = torch.matmul(synthetic_x, synthetic_w) + synthetic_b
    # 算出一系列的张量，得到的 y 的形状应该与 x 是相同的，此时得到的是 num_examples 长的行向量
    synthetic_y += torch.normal(0, 0.01, synthetic_y.shape)  # 加上一个正态分布的随机的偏移量
    return synthetic_x, synthetic_y.reshape((-1, 1))  # y 是行向量，因此这里需要把 y 重塑成一个列向量输出


length_of_x = 3  # 线性模型的自变量个数

true_w = torch.normal(0, 5, (length_of_x, 1))  # 人造一个 w 参数
true_b = torch.normal(0, 5, (1, 1))  # 人造一个 b 参数
features, labels = synthetic_data(true_w, true_b, 1000)  # 生成 1000 组数据，将张量 x 赋值给 features，重塑成列向量后的 y 向量赋值给 label


# 定义一个 data_iter 函数用于生成小批量。该函数接收批量大小 batch_size、特征矩阵 features 和标签向量 labels 作为输入，生成大小为 batch_size 的小批量
def data_iter(data_iter_batch_size, data_iter_features, data_iter_labels):
    num_examples = len(data_iter_features)  # data_iter_features 就是 x 张量，其第一个维度长度就是样本的个数，即 num_examples
    indices = list(range(num_examples))  # 对应创建一个有这么多长度下标的 list
    random.shuffle(indices)  # 这些样本是随机读取的，没有特定的顺序，因此在这里将其打乱
    for i in range(0, num_examples, data_iter_batch_size):  # 以 batch_size 为步长进行跳转，选取出一组长为 batch_size 的样本
        batch_indices = torch.tensor(indices[i:min(i + data_iter_batch_size, num_examples)])
        # min 函数是用来防止步长循环出现越界的情况。把从 i 到 i + batch_size 这一步之间的下标转成一个张量，再赋值给batch_indices
        yield data_iter_features[batch_indices], data_iter_labels[batch_indices]  # 产生随机顺序的特征及其标号，返回一次


# 设置超参数 batch_size 为 10
batch_size = 10

for x, y in data_iter(batch_size, features, labels):  # 相当于反复调用函数
    # 输出十个样本数据
    print(x, '\n', y)
    break

# 以上数据已经准备好了，接下来开始初始化参数模型。由于需要计算梯度，因此 requires_grad 均为 True
w = torch.normal(0, 0.01, size=(length_of_x, 1), requires_grad=True)  # w 是一个长为 2 的一个向量，将其初始化为均值为 0，方差为 0.01 的正态分布向量
b = torch.zeros(1, requires_grad=True)  # 对于偏差 b 来说，就是一个长度为 1，初始值为 0 的标量


def linreg(linreg_x, linreg_w, linreg_b):  # 接下来定义线性回归模型
    """线性回归模型。"""
    return torch.matmul(linreg_x, linreg_w) + linreg_b


def square_loss(y_hat, square_loss_y):  # 定义损失函数，这里选取的是均方误差函数
    """均方损失"""
    return (y_hat - square_loss_y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, sgd_lr, sgd_batch_size):  # 定义优化算法，其中 params 是包含了各种参数的一个 list，里面包括了 w 和 b，然后在给定学习率 lr 和批大小 batch_size
    """小批量随机梯度下降"""
    with torch.no_grad():  # 不需要梯度计算
        for param in params:  # 对于 list 中的每一个参数
            # 由于沿梯度方向会增加损失函数值，因此这里将参数加上它的负梯度，使得参数趋向于损失函数尽可能小的方向
            param -= sgd_lr * param.grad / sgd_batch_size  # 之前的损失函数没有求均值，因此这里需要求均值
            # 通过除以 sgd_batch_size 可以避免批量大小影响步长
            param.grad.zero_()  # 完成以后，手动将梯度设置为零


# 下面开始进行训练，指定一些超参数


lr = 0.03  # 设置学习率
num_epochs = 3  # 设置迭代次数
net = linreg  # 选用线性回归模型
loss = square_loss  # 选用均方损失函数

for epoch in range(num_epochs):  # 迭代循环
    for X, y in data_iter(batch_size, features, labels):  # 每次从数据集中拿出批量大小的 X 和 Y
        loss_value = loss(net(X, w, b), y)  # 把 X, w, b 放进对应的模型中，衡量与 y 的实际值之间的 损失
        # 因为 l 形状是 ('batch_size', 1)，而不是一个标量。`1` 中的所有元素被加到
        # 并以此计算关于 ['w', 'b'] 的梯度
        loss_value.sum().backward()  # 先求和，再计算梯度。此时 param.grad 存储的就是损失函数关于参数矩阵 param 的梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新
    with torch.no_grad():  # 这部分不需要计算梯度，因此放在这里
        train_l = loss(net(features, w, b), labels)  # 把整个数据 features 放进去计算，再和真实的 label 做一下损失，用于衡量模型的效果
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print("true_w=", true_w, "true_b=", true_b, "\nlearned_w=", w, "learned_b=", b)
