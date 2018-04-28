import tensorflow as tf
from config import *

from tf_TextRNN import TextRNN
import datetime

df = pd.read_csv(input_file, encoding="utf-8")

question1 = df['question1'].values
question2 = df['question2'].values
y = df['label'].values

def train(x_train1, x_train2, y_train, x_val1, x_val2, y_val, model=TextRNN(), config=TRNNConfig()):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    out_dir = 'textrnn'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(1e-3)
    train_step = optimizer.minimize(model.loss)
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

    def train_step(x_batch1, x_batch2, y_batch):
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
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch1,x_batch2, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            model.input_x1: x_batch1,
            model.input_x2: x_batch2,
            model.input_y: y_batch,
            model.keep_prob: 1.0
        }
        step, summaries, loss, accuracy = session.run(
            [global_step, dev_summary_op, model.loss, model.acc],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("dev {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return loss, accuracy

    def batch_iter(x1, x2, y, batch_size):
        idx = np.arange(len(y))
        batches = [idx[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
                   range(len(y) // batch_size + 1)]
        for i in batches:
            yield x1[i], x2[i], y[i]

    best_acc_val = 0
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        total_batch = 0
        for x_batch1, x_batch2, y_batch in batch_iter(x_train1, x_train2, y_train, config.batch_size):
            train_step(x_batch1, x_batch2, y_batch)
            total_batch += 1
        if epoch % config.print_per_batch == 0:
            # 每多少轮次输出在训练集和验证集上的性能
            loss_val, acc_val = dev_step(x_val1, x_val2, y_val, writer=dev_summary_writer)  # todo

            if acc_val > best_acc_val:
                # 保存最好结果
                best_acc_val = acc_val
                path = saver.save(sess=session, save_path=checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))



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
    x_val_q1 = X_train_q1[ind_te]
    x_val_q2 = X_train_q2[ind_te]
    y_train = y[ind_tr]
    y_val = y[ind_te]
    # mymodel = TextRNN()
    train(x_train1= x_train_q1, x_train2= x_train_q2,y_train=y_train,
          x_val1= x_val_q1, x_val2= x_val_q2, y_val=y_val)
    break
