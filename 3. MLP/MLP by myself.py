import torch
from torch import nn

net = nn.Sequential(nn.Flatten(), nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

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
