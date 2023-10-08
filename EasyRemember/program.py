# 此 py 文件描述了 EasyRemember 的基本工作原理，但不包含机器学习内容

import time
import torch
import numpy
from torch import nn

# 超参数设置
Length_of_memory = 6  # 对于一张卡片来说，程序只会记住最后 length_of_memory 次卡片的出现。
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
Batch_size = 10

# 初始化模型参数
num_hidden = 10  # 隐藏层维度
w1 = nn.Parameter(torch.randn(Length_of_memory * 3 - 1, num_hidden, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hidden, 1, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(1, requires_grad=True))


class Card:
    def __init__(self):
        self.front = ''
        self.back = ''
        self.time = 0  # 单位为 ms
        self.history = []
        self.react = 0
        self.expect_show_up = 0  # 单位为秒，这里是绝对时间戳
        self.true_show_up = 0  # 猜测出来的"真实出现时间"，用于计算损失函数

    def show_front(self):
        print('\n\n\n\n')
        print(self.front)
        self.time = time.time()
        input('\n\n按回车出示背面...')
        self.time = self.time - time.time()
        print('\n\n\n\n\n\n')

    def show_back(self):
        print('\n\n\n\n\n\n')
        print(self.front, '\n')
        print(self.back)

        # try:
        self.react = int(input('请选择难易程度:\n1:不会, 2: 困难, 3: 记得, 4: 熟练\n'))
        '''except ValueError:
            print('输入不正确，请重新输入！')
            self.react = int(input('请选择难易程度:\n1:不会, 2: 困难, 3: 记得, 4: 熟练\n'))'''
        self.history.append([self.time, self.react, 0])  # 补个零
        if len(self.history) > Length_of_memory - 1:
            self.history.pop(0)  # 防止过长
        self.calculate_expect_show_up()
        print('此卡片将在', self.expect_show_up - time.time(), '后出现。\n\n\n')
        self.history[-1].append(self.expect_show_up - time.time())  # 记录的是相对时间


    def calculate_expect_show_up(self):
        # self.expect_show_up = time.time() + self.react * 60 - self.time * 6  # 默认算法
        self.true_show_up = self.history[-2][2] + (self.react - 2.5) ** 3 / self.time
        self.expect_show_up = net(self.history)  # 深度学习算法
        # 计算损失函数
        loss = (self.expect_show_up - self.true_show_up) ** 2
        loss.backward()
        sgd([w1, b1, w2, b2], lr, Batch_size)
        with torch.no_grad():
            print('损失为', int(loss))

    def edit(self):
        choice = input('1: 修改正面\n2: 修改反面')
        if choice == '1':
            print("将", self.front, '改为：')
            self.front = input()
        if choice == '2':
            print("将", self.back, '改为：')
            self.back = input()


def read_card(card_list):
    try:
        f = open('card.txt', 'r', encoding='utf-8')
    except FileNotFoundError:
        open('card.txt', 'w')
        f = open('card.txt', 'r', encoding='utf-8')

    lines = f.readline()
    counter = 0
    # 初始化测试卡片
    while lines:
        parameter = lines.split(",")
        print(lines)
        card_list.append(Card())
        card_list[counter].front = parameter[0]
        card_list[counter].back = parameter[1]
        card_list[counter].expect_show_up = float(parameter[2])
        for j_counter in range((len(parameter) - 3) // 3):
            card_list[counter].history.append([int(parameter[3 + 3 * j_counter]), int(parameter[4 + 3 * j_counter]),
                                              int(parameter[5 + 3 * j_counter])])
        lines = f.readline()
        counter += 1
    f.close()


def save_card(card_list):
    f = open('card1.txt', 'w', encoding='utf-8')
    for card in card_list:
        f.write(card.front)
        f.write(',')
        f.write(card.back)
        for i in card.history:
            f.write(',')
            f.write(i)
        f.write('\n')


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x):
    x = numpy.array(x)
    x = x.reshape(1, -1)
    x = numpy.delete(x, -1)
    x = torch.from_numpy(x).float()
    x.reshape(-1, Length_of_memory * 3-1)
    # 接下来定义隐藏层
    h = relu(x @ w1 + b1)
    return relu(h @ w2 + b2)


def sgd(params, learning_rate, batch_size):
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()


Card_List = []
read_card(Card_List)
while True:
    Card_List = sorted(Card_List, key=lambda x: x.expect_show_up)
    if Card_List[0].expect_show_up - time.time() <= 3600:  # 最近的卡片将在一小时内出现
        Card_List[0].show_front()
        Card_List[0].show_back()
    else:
        print("未来 1 小时内没有到期的卡片，程序保存并退出")
        save_card(Card_List)
        break
