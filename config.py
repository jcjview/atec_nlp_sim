input_file = "./input/process.csv"
w2vpath = './data/baike.128.no_truncate.glove.txt'
embedding_matrix_path = './baseline/temp.npy'
kernel_name="bilstm"
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf

MAX_TEXT_LENGTH = 50
MAX_FEATURES = 10000
embedding_dims = 128
dr = 0.2
batch_size = 256

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = MAX_TEXT_LENGTH        # 序列长度
    num_classes = 1        # 类别数
    vocab_size = MAX_FEATURES       # 词汇表达小

    num_layers= 1           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru
    fc_hidden_dim=64
    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 256         # 每批训练大小
    num_epochs = 50          # 总迭代轮次
    early_stop=5

    print_per_batch = 1    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    num_checkpoints=5  #Number of checkpoints to store (default: 5)

    class_weight0=1.0
    class_weight1=2.3
