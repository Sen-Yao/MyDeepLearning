import torch
import torch.nn.functional as func
from torch import nn


# 加载和保存张量
x = torch.arange(4)  # 长为 4 的向量
torch.save(x, 'x-file')  # 将向量 x 存储在这个叫做 x-file 的文件里
x2 = torch.load("x-file")  # 访问并读取 x-file 的文件，加载到 x2 中

# 列表和字典同样可以存储和读取
y = torch.arange(4)
torch.save([x, y], 'x-file')
x2, y2 = torch.load("x-file")

mydict = {'x': x, 'y': y}
torch.save(mydict, 'x-file')
mydict2 = torch.load("x-file")


# 加载和保存模型参数
# 其实只需要存储权重就好了。比如给出一个最简单的 MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 继承父类
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 20)

    # nn.Module 类中，__call__ 方法会调用 forward 方法。这里重写 nn.Module 模块
    def forward(self, x):  # x 是输入
        # 先根据输入，求出隐藏层的输出；然后通过 ReLU 激活函数，最后通过 self.out 输出，这就完成了前向计算
        return self.out(func.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


# 现在就可以将模型的参数存储在一个叫做 mlp.params 的文件
torch.save(net.state_dict(), 'mlp.params')

# 如果要读取，则先需要实例化 MLP
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))  # 用文件中的参数覆盖掉初始化参数
