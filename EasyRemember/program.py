import time


class Card:
    def __init__(self):
        self.front = ''
        self.back = ''
        self.time = 0  # 单位为 ms
        self.history = []
        self.react = 0
        self.expect_show_up = 0  # 单位为秒

    def show_front(self):
        print('\n\n\n\n\n\n')
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
        self.expect_show_up = time.time() + self.react * 60
        self.history.append([self.time, self.react, self.expect_show_up])
        if len(self.history) > 10:
            self.history.pop(0)  # 防止过长
        print('\n\n\n\n\n\n')

    def edit(self):
        choice = input('1: 修改正面\n2: 修改反面')
        if choice == '1':
            print("将", self.front, '改为：')
            self.front = input()
        if choice == '2':
            print("将", self.back, '改为：')
            self.back = input()


try:
    f = open('card.txt', 'r')
except FileNotFoundError:
    open('card.txt', 'w')
    f = open('card.txt', 'r')

card_list = []
lines = f.readline()
counter = 0
# 初始化测试卡片
while lines:
    parameter = lines[counter].split(",")
    print(lines)
    card_list.append(Card())
    card_list[counter].front = parameter[0]
    card_list[counter].back = parameter[1]
    card_list[counter].expect_show_up = parameter[2]
    for j_counter in range(len(lines[counter])):
        card_list[counter].history[j_counter // 3][j_counter % 3] = parameter[3+j_counter]
    lines = f.readline()
    counter += 1
f.close()
while True:
    card_list = sorted(card_list, key=lambda x: x.expect_show_up)
    card_list[0].show_front()
    card_list[0].show_back()
