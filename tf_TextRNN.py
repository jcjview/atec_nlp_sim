import tensorflow as tf
from config import *


class TextRNN():
    def __init__(self,
                 embedding_matrix=None,
                 config=TRNNConfig()):
        self.config = config

        def lstm_cell():  # lstm核
            cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_dim, forget_bias=0.0, state_is_tuple=True)
            if config.dropout_keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=config.dropout_keep_prob
                )
            return cell

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)



        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x2')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 词向量映射
        with tf.device('/cpu:0'):
            # weW = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            weW = tf.get_variable(name="W", shape=embedding_matrix.shape, initializer=tf.constant_initializer(embedding_matrix),trainable=True)
            embedding_inputs1 = tf.nn.embedding_lookup(weW, self.input_x1)
            embedding_inputs2 = tf.nn.embedding_lookup(weW, self.input_x2)
            print('input_x1', self.input_x1.get_shape())
        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [lstm_cell() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs1, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs1, dtype=tf.float32)
            _outputs2, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs2, dtype=tf.float32)
            print("_outputs2", _outputs2.get_shape())
            encode1 = _outputs1[:, -1, :]
            encode2 = _outputs2[:, -1, :]  # 取最后一个时序输出作为结果
            print("encode2", encode2.get_shape())
            last = tf.multiply(encode1, encode2, name="last")
            print("multiply",last.get_shape())
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.fc_hidden_dim, name='fc1',activation=tf.nn.relu)
            # fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # fc = tf.nn.relu(fc)
            print('fc',fc.get_shape())
            # 分类器
            # lbW = tf.Variable(tf.truncated_normal([self.config.hidden_dim, self.config.num_classes], stddev=0.1), name="lbW")
            # b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name="b")
            # print('lbW',lbW.get_shape())
            # print('b',b.get_shape())
            self.scores = tf.layers.dense(fc,1,activation=tf.nn.sigmoid)  # Softmax
            # self.scores = tf.nn.xw_plus_b(fc, lbW, b, name="scores")
            self.y_pred_cls = tf.round(self.scores, name="predictions")
            # self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,labels=self.input_y)
            # self.loss = tf.reduce_mean(cross_entropy)

            # self.loss = -tf.reduce_sum(tf.cast(self.input_y, tf.float32)
            #                                           * tf.log(tf.cast(self.y_pred_cls, tf.float32)), reduction_indices=1)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.input_y* tf.log(self.scores)*config.class_weight1
                                                      +(1-self.input_y)*tf.log(1-self.scores)*config.class_weight0
                                                      , reduction_indices=[1]))

            # self.loss = tf.losses.sigmoid_cross_entropy(logits=self.scores, multi_class_labels=self.input_y)
            # 优化器
            # self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.cast(self.input_y, tf.float32), tf.cast(self.y_pred_cls, tf.float32))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
