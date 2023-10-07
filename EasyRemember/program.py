# 此 py 文件描述了 EasyRemember 的基本工作原理，但不包含机器学习内容

import time
import torch
from torch import nn

# 超参数设置
length_of_memory = 5  # 对于一张卡片来说，程序只会记住最后 length_of_memory 次卡片的出现。
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
net = nn.Sequential(nn.Linear(length_of_memory, 1))
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)


class Card:
    def __init__(self):
        self.front = ''
        self.back = ''
        self.time = 0  # 单位为 ms
        self.history = []
        self.react = 0
        self.expect_show_up = 0  # 单位为秒

    def show_front(self):
        print('\n\n\n\n')
        print(self.front)
        self.time = time.time()
        input('\n\n按回车出示背面...')
        print('\n\n\n\n\n\n')

    def show_back(self):
        print('\n\n\n\n\n\n')
        print(self.front, '\n')
        print(self.back)
        self.time = self.time - time.time()
        try:
            self.react = int(input('请选择难易程度:\n1:不会, 2: 困难, 3: 记得, 4: 熟练\n'))
        except ValueError:
            print('输入不正确，请重新输入！')
            self.react = int(input('请选择难易程度:\n1:不会, 2: 困难, 3: 记得, 4: 熟练\n'))
        self.calculate_expect_show_up()
        self.history.append([self.time, self.react, self.expect_show_up])
        if len(self.history) > length_of_memory:
            self.history.pop(0)  # 防止过长
        print('此卡片将在', self.expect_show_up - time.time(), '后出现。\n\n\n')

    def calculate_expect_show_up(self):
        self.expect_show_up = time.time() + self.react * 60 - self.time * 6
        w = torch.normal(0, 0.01, size=(length_of_memory, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)


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
        for j_counter in range((len(parameter) - 3) / 3):
            card_list[counter].history[0].append(parameter[3 + 3 * j_counter])
            card_list[counter].history[1].append(parameter[4 + 3 * j_counter])
            card_list[counter].history[2].append(parameter[5 + 3 * j_counter])
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
