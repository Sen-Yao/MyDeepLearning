import random
import torch
import time
from torch import nn

# 初始化模型参数
Length_of_memory = 6
num_hidden = 10
W1 = nn.Parameter(torch.randn(Length_of_memory, num_hidden, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden, 1, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(1, requires_grad=True))

params = [W1, b1, W2, b2]

Data_list = []


def synthetic_data(length_of_memory, example_num=10):
    f = open('test.txt', "w", encoding='utf-8')
    # 此函数生成一个长度为 2 * length_of_memory + 2 的 list，模拟真人的反馈。
    # 我们首先生成一个熟练度猜测，这是一个 (50,50) 的正态分布。
    for counter in range(example_num):
        f.write('正面,反面,')
        f.write(str(time.time() + random.randint(0, 100)))
        f.write(',')
        synthetic_sequential = []
        proficiency_score = random.normalvariate(50, 20)
        if proficiency_score < 0:
            proficiency_score = 0.1
        # 对于某一张卡片，进行循环。
        for memory_counter in range(length_of_memory):
            if memory_counter == 0:
                card_time = int(50000 / proficiency_score) + random.randint(-200, 200)
                react = int(proficiency_score / 25 + 1)
                if react > 4:
                    react = 4
                if react < 1:
                    react = 1
                expect = int(1800 * proficiency_score)
                synthetic_sequential.append([card_time, react, expect])
            else:
                lucky_num = random.randint(1, 12)  # 对于每次出现，有时候运气不好刚好忘了，模拟这个因素
                # print('lucky=', lucky_num)
                if lucky_num < 5:
                    card_time = int(synthetic_sequential[-1][0] + (lucky_num - 6) * 200) + random.randint(-100, 100)
                    react = lucky_num // 2
                    expect = int(synthetic_sequential[-1][2] * 0.05 * react ** 2 + (react - 4) * 5000
                                 + random.randint(-2000, 2000))
                    if react > 4:
                        react = 4
                    if react < 1:
                        react = 1
                    if card_time < 100:
                        card_time = int(random.random() * 200) + 200
                    if expect < 0:
                        expect = 300
                    synthetic_sequential.append([card_time, react, expect])
                if 5 <= lucky_num < 9:
                    card_time = int(synthetic_sequential[-1][0] + (6 - lucky_num) * 1000 + random.randint(-200, 200))
                    react = lucky_num // 2
                    expect = int(synthetic_sequential[-1][2] + (react - 2.5) * 10000 + random.randint(-2000, 2000))
                    if react > 4:
                        react = 4
                    if react < 1:
                        react = 1
                    if card_time < 100:
                        card_time = int(random.random() * 200) + 200
                    if expect < 0:
                        expect = 300
                    synthetic_sequential.append([card_time, react, expect])
                if lucky_num >= 9:
                    card_time = int(synthetic_sequential[-1][0] * 1.5 + (9 - lucky_num) * 200) + random.randint(-100, 100)
                    react = lucky_num // 2
                    expect = int(synthetic_sequential[-1][2] * 0.5 * react ** 2 + random.randint(-2000, 2000))
                    if react > 4:
                        react = 4
                    if react < 1:
                        react = 1
                    if card_time < 100:
                        card_time = int(random.random() * 200) + 200
                    if expect < 0:
                        expect = 300
                    synthetic_sequential.append([card_time, react, expect])
            # print(synthetic_sequential[-1])  #dui
        '''for i in range(length_of_memory):
            print(synthetic_sequential[-memory_counter+i])'''
        # 接下来将数据存储到文件中
        for i in range(len(synthetic_sequential)):
            f.write(str(synthetic_sequential[i][0]))
            f.write(',')
            f.write(str(synthetic_sequential[i][1]))
            f.write(',')
            f.write(str(synthetic_sequential[i][2]))
            f.write(',')
        f.write('\n')
        Data_list.append(synthetic_sequential)
    f.close()


def data_iter(batch_size, data_list):
    num_examples = len(data_list)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i+batch_size, num_examples)])
        yield data_list[batch_indices]


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def net(x):
    x = x.reshape(-1, Length_of_memory*3)
    # 接下来定义隐藏层
    h = relu(x@W1 + b1)
    return relu(h@W2 + b2)


synthetic_data(Length_of_memory)
