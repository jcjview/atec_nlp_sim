input_file = "../input/process.csv"
w2vpath = '../data/baike.128.truncate.glove.txt'
embedding_matrix_path = './temp.npy'
kernel_name="lstm"
import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

MAX_TEXT_LENGTH = 50
MAX_FEATURES = 10000
embedding_dims = 128
dr = 0.2


def get_model(embedding_matrix,nb_words):
    input1_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    input2_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=True)
    seq_embedding_layer = keras.layers.LSTM(256, activation='tanh',recurrent_dropout=dr)
    seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))
    merge_layer = keras.layers.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])
    dense1_layer = keras.layers.Dense(16, activation='relu')(merge_layer)
    ouput_layer = keras.layers.Dense(1, activation='sigmoid')(dense1_layer)
    model = keras.models.Model([input1_tensor, input2_tensor], ouput_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()
    return model


class F1ScoreCallback(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(F1ScoreCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('f1_score' in self.params['metrics']):
            self.params['metrics'].append('f1_score')
        if not ('recall' in self.params['metrics']):
            self.params['metrics'].append('recall')

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['f1_score'] = float('0')
        logs['recall'] = float('0')
        if (self.validation_data):
            y_predict = self.model.predict([self.validation_data[0], self.validation_data[1]],
                                           batch_size=self.predict_batch_size)
            y_predict = (y_predict > 0.5).astype(int)
            logs['recall'] = accuracy_score(self.validation_data[2], y_predict)
            logs['f1_score'] = recall_score(self.validation_data[2], y_predict)
            print("recall f1_score", logs['recall'], logs['f1_score'])


from tqdm import tqdm
import mmap
import os


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_embedding_matrix(word_index, Emed_path, Embed_npy):
    if (os.path.exists(Embed_npy)):
        return np.load(Embed_npy)
    print('Indexing word vectors')
    embeddings_index = {}
    file_line = get_num_lines(Emed_path)
    print('lines ', file_line)
    with open(Emed_path, encoding='utf-8') as f:
        for line in tqdm(f, total=file_line):
            values = line.split()
            if(len(values)<128):
                print(values)
                continue
            word = ' '.join(values[:-128])
            coefs = np.asarray(values[-128:], dtype='float32')
            embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    print('Preparing embedding matrix')
    nb_words = MAX_FEATURES#min(MAX_FEATURES, len(word_index))
    all_embs = np.stack(embeddings_index.values())
    print(all_embs.shape)
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embedding_matrix = np.random.normal(loc=emb_mean, scale=emb_std, size=(nb_words, embedding_dims))

    # embedding_matrix = np.zeros((nb_words, embedding_dims))
    count=0
    for word, i in tqdm(word_index.items()):
        if i >= MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count+=1
    np.save(Embed_npy, embedding_matrix)
    print('Null word embeddings: %d' % (nb_words-count))
    print('not Null word embeddings: %d' % count)
    print('embedding_matrix shape', embedding_matrix.shape)
    # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


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
embedding_matrix1 = get_embedding_matrix(tokenizer.word_index, w2vpath, embedding_matrix_path)
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
    model = get_model(embedding_matrix1,nb_words)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    bst_model_path =kernel_name+'_weight_%d.h5' % count
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
                                       save_best_only=True, verbose=1, save_weights_only=True)
    hist = model.fit([x_train_q1,x_train_q2], y_train,
                     validation_data=([x_val_q1,x_val_q2], y_val),
                     epochs=5, batch_size=256, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint])
    model.load_weights(bst_model_path)
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
