import torch
from torch import nn
from torch.nn import functional as func

X = torch.rand(2, 4)
# X = torch.rand(size=(2, 20))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 256)
        self.out = nn.Linear(256, 4)

    def forward(self, x):  # x 是输入
        return self.out(func.relu(self.hidden(x)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block  # pytorch 需要的层，本身作为键值

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


net = MySequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 4))
print("x=", X, '\n')
# 在 nn.Module 类中，__call__ 方法会调用 forward 方法。因此，执行 net(X) 实际上是在调用 forward 方法，并将输入 X 作为参数传递给 forward 方法
print("net(x)=", net(X), '\n')

