import tensorflow as tf
from config import *


class TextRNN():
    def __init__(self,
                 embedding_matrix=None,
                 config=TRNNConfig()):
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
        self.input_x1 = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x2')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 词向量映射
        with tf.device('/cpu:0'):
            weW = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs1 = tf.nn.embedding_lookup(weW, self.input_x1)
            embedding_inputs2 = tf.nn.embedding_lookup(weW, self.input_x2)
            # W = tf.get_variable(name="W", shape=embedding_matrix.shape, initializer=tf.constant_initializer(embedding_matrix),trainable=True)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs1, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs1, dtype=tf.float32)
            _outputs2, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs2, dtype=tf.float32)
            encode1 = _outputs1[:, -1, :]
            encode2 = _outputs2[:, -1, :]  # 取最后一个时序输出作为结果
            last = tf.multiply(encode1, encode2, name="last")
            print(last.get_shape())
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            print('fc',fc.get_shape())
            # 分类器
            lbW = tf.Variable(tf.truncated_normal([self.config.hidden_dim, self.config.num_classes], stddev=0.1), name="lbW")
            b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            print('lbW',lbW.get_shape())
            print('b',b.get_shape())
            self.scores = tf.nn.xw_plus_b(fc, lbW, b, name="scores")
            self.y_pred_cls = tf.round(self.scores, name="predictions")
            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.input_y)
            # self.loss = tf.reduce_mean(cross_entropy)

            # self.loss = -tf.reduce_sum(tf.cast(self.input_y, tf.float32)
            #                                           * tf.log(tf.cast(self.y_pred_cls, tf.float32)), reduction_indices=1)
            print('scores', self.scores)
            print('input_y', self.input_y)
            self.loss = tf.losses.sigmoid_cross_entropy(logits=self.scores, multi_class_labels=self.input_y)
            # 优化器
            # self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.cast(self.input_y, tf.float32), tf.cast(self.y_pred_cls, tf.float32))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
