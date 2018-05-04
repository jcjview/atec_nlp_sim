input_file = "./input/process.csv"
w2vpath = './data/baike.128.no_truncate.glove.txt'
embedding_matrix_path = './baseline/temp.npy'
kernel_name="bilstm"
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import tensorflow as tf
import datetime
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

df = pd.read_csv(input_file, encoding="utf-8")

question1 = df['question1'].values
question2 = df['question2'].values
y = df['label'].values
y=np.array(y,dtype=np.float32)
embedding_matrix1=np.load(embedding_matrix_path)
def train(x_train1, x_train2, y_train, x_val1, x_val2, y_val, model=TextRNN(embedding_matrix=embedding_matrix1), config=TRNNConfig()):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    out_dir = 'textrnn'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # optimizer = tf.train.GradientDescentOptimizer(5e-3)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_step_ = optimizer.minimize(model.loss)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.acc)
    # Keep track of gradient values and sparsity (optional)
    # grad_summaries = []
    # for g, v in grads_and_vars:
    #     if g is not None:
    #         grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
    #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
    #         grad_summaries.append(grad_hist_summary)
    #         grad_summaries.append(sparsity_summary)
    # grad_summaries_merged = tf.summary.merge(grad_summaries)
    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints123"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

    def train_fun(x_batch1, x_batch2, y_batch):
        """
        A single training step
        """
        feed_dict = {
            model.input_x1: x_batch1,
            model.input_x2: x_batch2,
            model.input_y: y_batch,
            model.keep_prob: config.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = session.run(
            [train_op, global_step, train_summary_op, model.loss, model.acc],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_fun(x_batch1,x_batch2, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            model.input_x1: x_batch1,
            model.input_x2: x_batch2,
            model.input_y: y_batch,
            model.keep_prob: 1.0
        }
        step, summaries, loss, accuracy,predict = session.run(
            [global_step, dev_summary_op, model.loss, model.acc,model.scores],
            feed_dict)
        pred_label = (predict > 0.5).astype(int)
        print(np,sum(pred_label),np,sum(predict))
        accuracy1 = accuracy_score(y_batch, pred_label)
        recall = recall_score(y_batch, pred_label)
        precision = precision_score(y_batch, pred_label)
        time_str = datetime.datetime.now().isoformat()
        print("dev {}: step {}, loss {:g}, acc {:g},acc1 {:g},recall {:g},precision {:g}".format(time_str, step, loss, accuracy,accuracy1,recall,precision))
        if writer:
            writer.add_summary(summaries, step)
        return loss, accuracy,predict

    def batch_iter(x1, x2, y, batch_size):
        idx = np.arange(len(y))
        batches = [idx[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
                   range(len(y) // batch_size + 1)]
        for i in batches:
            yield x1[i], x2[i], y[i]

    best_acc_val = 0
    monitor_early_stop=0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        total_batch = 0
        for x_batch1, x_batch2, y_batch in batch_iter(x_train1, x_train2, y_train, config.batch_size):
            train_fun(x_batch1, x_batch2, y_batch)
            total_batch += 1
        if epoch % config.print_per_batch == 0:
            # 每多少轮次输出在训练集和验证集上的性能
            loss_val, acc_val,predict = dev_fun(x_val1, x_val2, y_val, writer=dev_summary_writer)  # todo

            if acc_val > best_acc_val:
                # 保存最好结果
                best_acc_val = acc_val
                path = saver.save(sess=session, save_path=checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))
                monitor_early_stop=0
            else:
                monitor_early_stop+=1
                print("do not save ")
                if(monitor_early_stop>=config.early_stop):
                    break
    loss_val, acc_val, predict = dev_fun(x_val1, x_val2, y_val)
    return predict


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(question1) + list(question2))
list_tokenized_question1 = tokenizer.texts_to_sequences(question1)
list_tokenized_question2 = tokenizer.texts_to_sequences(question2)
X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=MAX_TEXT_LENGTH)
X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=MAX_TEXT_LENGTH)
seed = 20180426
cv_folds = 10
y=np.reshape(y,[len(y),1])
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
pred_oob = np.zeros(shape=(len(y), 1))
# print(pred_oob.shape)
count = 0
for ind_tr, ind_te in skf.split(X_train_q1, y):
    x_train_q1 = X_train_q1[ind_tr]
    x_train_q2 = X_train_q2[ind_tr]
    y_train = y[ind_tr]

    x_val_q1 = X_train_q1[ind_te]
    x_val_q2 = X_train_q2[ind_te]
    y_val = y[ind_te]
    # mymodel = TextRNN()
    predict=train(x_train1= x_train_q1, x_train2= x_train_q2,y_train=y_train,
          x_val1= x_val_q1, x_val2= x_val_q2, y_val=y_val)
    pred_oob[ind_te]=predict
    # break
pred_label = (pred_oob > 0.5).astype(int)
recall = recall_score(y, pred_label)
print("recal", recall)
precision = precision_score(y, pred_label)
print("precision", precision)
accuracy = accuracy_score(y, pred_label)
print("accuracy", accuracy)
f1 = f1_score(y, pred_label)
print("f1", f1)