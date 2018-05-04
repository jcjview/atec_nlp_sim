import tensorflow as tf
from config import *

filter_sizes=[2,3,8,9]
num_filters=3
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
            W = tf.Variable(
                tf.random_uniform([config.vocab_size, config.embedding_dim], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # W = tf.get_variable(name="W", shape=embedding_matrix.shape, initializer=tf.constant_initializer(embedding_matrix),trainable=True)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, config.embedding_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.vocab_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.config.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, config.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.input_y)
            # self.loss = tf.reduce_mean(cross_entropy)

            # self.loss = -tf.reduce_sum(tf.cast(self.input_y, tf.float32)
            #                                           * tf.log(tf.cast(self.y_pred_cls, tf.float32)), reduction_indices=1)
            self.loss=tf.losses.mean_squared_error(logits=self.scores,labels=self.input_y)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.cast(self.input_y, tf.float32), tf.cast(self.y_pred_cls, tf.float32))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
