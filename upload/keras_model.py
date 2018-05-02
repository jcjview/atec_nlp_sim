#/usr/bin/env python
#coding=utf-8

import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

MAX_TEXT_LENGTH = 50
MAX_FEATURES = 10000
embedding_dims = 128
dr = 0.0
embedding_matrix_path="./temp.npy"
input_file = "process.csv"
kernel_name = "bilstm"
def get_model(embedding_matrix):
    input1_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    input2_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   weights=[embedding_matrix],
                                                   input_length=MAX_TEXT_LENGTH,
                                                   trainable=True)
    seq_embedding_layer = keras.layers.Bidirectional(keras.layers.GRU(256, recurrent_dropout=dr))
    seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))
    merge_layer = keras.layers.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])
    merge_layer = keras.layers.Dropout(dr)(merge_layer)
    dense1_layer = keras.layers.Dense(64, activation='relu')(merge_layer)
    ouput_layer = keras.layers.Dense(1, activation='sigmoid')(dense1_layer)
    model = keras.models.Model([input1_tensor, input2_tensor], ouput_layer)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    # model.summary()
    return model
def process(inpath):
    q1 = []
    q2 = []
    with open(inpath, 'r') as fin:
        for line in fin:
            lineno, sen1, sen2 = line.strip(",").split('\t')
            q1.append(sen1)
            q2.append(sen2)
    return q1,q2

question1,question2 = process(input_file)
embedding_matrix1 = np.load(embedding_matrix_path)
cv_folds = 10

class keras_model:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_FEATURES)
        self.tokenizer.fit_on_texts(list(question1) + list(question2))
        pass
    def predict(self,d1,d2):
        pred_test = np.zeros(shape=(len(d1), 1))
        count = 0
        list_tokenized_question1 = self.tokenizer.texts_to_sequences(d1)
        list_tokenized_question2 = self.tokenizer.texts_to_sequences(d2)
        x_val_q1 = pad_sequences(list_tokenized_question1, maxlen=MAX_TEXT_LENGTH)
        x_val_q2 = pad_sequences(list_tokenized_question2, maxlen=MAX_TEXT_LENGTH)
        for index in range(cv_folds):
            model = get_model(embedding_matrix1)
            bst_model_path = kernel_name + '_weight_%d.h5' % count
            model.load_weights(bst_model_path)
            y_predict = model.predict([x_val_q1, x_val_q2], batch_size=1024, verbose=1)
            pred_test+= y_predict
        pred_test /= cv_folds
        labels=(pred_test > 0.5).astype(int)
        return labels