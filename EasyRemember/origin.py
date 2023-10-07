import random
import torch

# 为了执行之后的操作，这里设计一个人造模型来生成数据。
# 根据带有噪声的线性模型构造一个人造数据集
# 我们使用线性模型参数列向量 w，标量 b 和噪声 epsilon 生成数据集及其标签
# w 表示人造模型的参数列向量，b 表示人造模型的偏移量列向量，num_examples 表示需要生成的样本数量


example_num = 3


def synthetic_data():
    # 对于记忆数据模拟，我们首先生成一个熟练度猜测，这是一个 (50,50) 的正态分布。
    proficiency_score = torch.normal(50, 50, (example_num, 1))
    synthetic_sequential = []
    for counter in range(example_num):
        synthetic_sequential.append([random.randint(1, 4), random.randint(1, 10000), 0])

