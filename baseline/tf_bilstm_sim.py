#!/usr/bin/python
# -*- coding: utf-8 -*-
#py2
from __future__ import  print_function
input_file = "../input/process.csv"
w2vpath = '../data/baike.128.no_truncate.glove.txt'
embedding_matrix_path = './temp_no_truncate.npy'
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
save_dir = 'checkpoints/textrnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = MAX_TEXT_LENGTH        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = MAX_FEATURES       # 词汇表达小

    num_layers= 1           # 隐藏层层数
    hidden_dim = 256        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 256         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

class TextRNN():
    def __init__(self,
                 embedding_matrix=None,
                 config=TRNNConfig):
        self.config = config
        def lstm_cell():  # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():  # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def train(model,config) :
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)
            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

df = pd.read_csv(input_file, encoding="utf-8")

question1 = df['question1'].values
question2 = df['question2'].values
y = df['label'].values
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(question1) + list(question2))
list_tokenized_question1 = tokenizer.texts_to_sequences(question1)
list_tokenized_question2 = tokenizer.texts_to_sequences(question2)
X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=MAX_TEXT_LENGTH)
X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=MAX_TEXT_LENGTH)
nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
print("nb_words",nb_words)
seed = 20180426
cv_folds = 10
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
pred_oob = np.zeros(shape=(len(y), 1))
# print(pred_oob.shape)
count = 0
for ind_tr, ind_te in skf.split(X_train_q1, y):
    x_train_q1 = X_train_q1[ind_tr]
    x_train_q2 = X_train_q2[ind_tr]
    x_val_q1 = X_train_q1[ind_te]
    x_val_q2 = X_train_q2[ind_te]
    y_train = y[ind_tr]
    y_val = y[ind_te]

    # model = get_model(embedding_matrix1,nb_words)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
    # bst_model_path =kernel_name+'_weight_%d.h5' % count
    # model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
    #                                    save_best_only=True, verbose=1, save_weights_only=True)
    # hist = model.fit([x_train_q1,x_train_q2], y_train,
    #                  validation_data=([x_val_q1,x_val_q2], y_val),
    #                  epochs=6, batch_size=256, shuffle=True,
    #                  class_weight={0: 1.3233, 1: 0.4472},
    #                  callbacks=[early_stopping, model_checkpoint])
    # model.load_weights(bst_model_path)
    y_predict = model.predict([x_val_q1, x_val_q2], batch_size=256, verbose=1)
    pred_oob[ind_te] = y_predict
    y_predict = (y_predict > 0.5).astype(int)
    recall = recall_score(y_val, y_predict)
    print(count, "recal", recall)
    precision = precision_score(y_val, y_predict)
    print(count, "precision", precision)
    accuracy = accuracy_score(y_val, y_predict)
    print(count, "accuracy ", accuracy)
    f1 = f1_score(y_val, y_predict)
    print(count, "f1", f1)
    count += 1
pred_oob = (pred_oob > 0.5).astype(int)
recall = recall_score(y, pred_oob)
print("recal", recall)
precision = precision_score(y, pred_oob)
print("precision", precision)
accuracy = accuracy_score(y, pred_oob)
print("accuracy", accuracy)
f1 = f1_score(y, pred_oob)
print("f1", f1)
