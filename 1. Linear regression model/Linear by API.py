import torch
from torch.utils import data  # 用于数据迭代器

# nn 是神经网络的缩写
from torch import nn


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


# 利用 torch 现有的 API 来构造一个数据迭代器

def load_array(data_arrays, load_array_batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    # DataLoader 函数每次可以从 dataset 中随机挑选 batch_size 个样本出来，若 shuffle 为 True，则它会将数据进行打乱
    # 对于训练的情况，自然是需要打乱的
    return data.DataLoader(dataset, load_array_batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 指定输入和输出维度，之后会详细介绍 Sequential，可以理解成是层的列表
net = nn.Sequential(nn.Linear(length_of_x, 1))

# 对模型参数进行初始化，下划线表示 Python 中的原位替换
# 其实 Linear 自带初始化，所以其实这里不需要写
# net[0].weight.data.normal_(0, 0.01)
# net[0].bias.data.fill_(0)

# 均方误差可以直接调用 nn 中的函数，在这里实例化为 loss
loss = nn.MSELoss()

# 对 SGD 进行实例化
# 需要至少两个输入，一个是网络的参数，即所有的 w 和 b，另一个是学习率作为超参数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 开始训练
num_epochs = 3  # 设置迭代次数

for epoch in range(num_epochs):  # 迭代循环
    for X, y in data_iter:  # 每次从数据集中拿出批量大小的 X 和 Y
        loss_value = loss(net(X), y)  # 把 X 放进对应的模型中衡量与 y 的实际值之间的 损失。由于此时 net 已经有了参数，因此不需要代入 w 和 b
        trainer.zero_grad()  # 将训练器的梯度清零
        loss_value.backward()  # 反向求导
        trainer.step()  # 调用 step 函数进行模型的更新
    loss_value = loss(net(features), labels)  # 重新计算损失函数
    print(f'epoch {epoch + 1}, loss {float(loss_value):f}')

print("\n\ntrue_w=", true_w.reshape(1, -1), "\nlearned_w=", net[0].weight.data, "\n\ntrue_b=", true_b, "\nlearned_b=",
      net[0].bias.data)
