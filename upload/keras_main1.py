# /usr/bin/env python
# coding=utf-8
input_file = "./train.txt"
embedding_matrix_path = './temp_no_truncate.npy'
kernel_name = "bilstm"
import numpy as np
import keras
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import jieba
import codecs

jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('余额宝')

MAX_TEXT_LENGTH = 50
MAX_FEATURES = 10000
embedding_dims = 128
dr = 0.01


def pandas_process(input_train):
    q1 = []
    q2 = []
    vlabel = []
    df = {}
    fin = codecs.open(input_train, 'r', encoding='utf-8')
    fin.readline()
    for line in fin:
        l, sen1, sen2 = line.strip().split('\t')
        q1.append(sen1)
        q2.append(sen2)
        vlabel.append(int(l))
    fin.close()
    df["question1"] = q1
    df["question2"] = q2
    df["label"] = vlabel
    return df


def seg(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)


def get_model():
    input1_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    input2_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,
                                                   # weights=[embedding_matrix],
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


if __name__ == '__main__':
    inpath = sys.argv[1]
    outputpath = sys.argv[2]
    # import pandas as pd
    # input_file = "../input/process.csv"
    # df=pd.read_csv(input_file)
    # question1 = df['question1'].values
    # question2 = df['question2'].values
    # y = df['label'].values
    df = pandas_process(input_file)
    question1 = df['question1']
    question2 = df['question2']
    y = df['label']
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer

    # np.savetxt('X_train_q1.out', X_train_q1, delimiter=',')
    # np.savetxt('X_train_q2.out', X_train_q2, delimiter=',')
    # inpath="test1.txt"
    test_data1 = []
    test_data2 = []
    linenos = []
    fin = codecs.open(inpath, 'r', encoding='utf-8')
    for line in fin:
        lineno, sen1, sen2 = line.strip().split('\t')
        sen1 = seg(sen1)
        sen2 = seg(sen2)
        test_data1.append(sen1)
        test_data2.append(sen2)
        linenos.append(lineno)
    fin.close()

    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(question1) + list(question2))
    list_tokenized_question1 = tokenizer.texts_to_sequences(question1)
    list_tokenized_question2 = tokenizer.texts_to_sequences(question2)
    X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=MAX_TEXT_LENGTH)
    X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=MAX_TEXT_LENGTH)
    list_tokenized_question11 = tokenizer.texts_to_sequences(test_data1)
    list_tokenized_question22 = tokenizer.texts_to_sequences(test_data2)
    x_val_q1 = pad_sequences(list_tokenized_question11, maxlen=MAX_TEXT_LENGTH)
    x_val_q2 = pad_sequences(list_tokenized_question22, maxlen=MAX_TEXT_LENGTH)

    # for i in range(len(x_val_q1)):
    #     t=np.array_equal(X_train_q1[i], x_val_q1[i])
    #     if not t:
    #         print X_train_q1[i]," | ",x_val_q1[i]
    #         print i,question1[i]," | ",test_data1[i]
    #     t=np.array_equal(X_train_q2[i], x_val_q2[i])
    #     if not t:
    #         print X_train_q2[i]," | ", x_val_q2[i]
    #         print i,question2[i]," | ",test_data2[i]

    nb_words = min(MAX_FEATURES, len(tokenizer.word_index))
    # print("nb_words", nb_words)
    # embedding_matrix1 = np.load(embedding_matrix_path)
    seed = 20180426
    cv_folds = 10
    # from sklearn.model_selection import StratifiedKFold

    # skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
    y = y[0:len(x_val_q1)]
    # print x_val_q1.shape
    pred_oob = np.zeros(shape=(len(x_val_q1), 1))
    # print pred_oob.shape
    count = 0
    # print "start to predict."
    model = get_model()
    for index in range(cv_folds):
        bst_model_path = kernel_name + '_weight_%d.h5' % count
        model.load_weights(bst_model_path)
        y_predict = model.predict([x_val_q1, x_val_q2], batch_size=1024, verbose=0)
        pred_oob += y_predict
        # print "*",
        # break
        # try:
        #     y_predict = (y_predict > 0.5).astype(int)
        #     recall = recall_score(y, y_predict)
        #     print(count, "recall", recall)
        #     precision = precision_score(y, y_predict)
        #     print(count, "precision", precision)
        #     accuracy = accuracy_score(y, y_predict)
        #     print(count, "accuracy ", accuracy)
        #     f1 = f1_score(y, y_predict)
        #     print(count, "f1", f1)
        #     count += 1
        # except:
        #     pass
    # print "predict done.Saving output to %s"%outputpath
    pred_oob /= cv_folds
    pred_oob1 = (pred_oob > 0.5).astype(int)
    fout = codecs.open(outputpath, 'w', encoding='utf-8')
    for index, la in enumerate(pred_oob1):
        lineno = linenos[index]
        fout.write(lineno + '\t%d\n' % la)
    # print "All is done."
            # try:
            #     recall = recall_score(y, pred_oob1)
            #     print("recal", recall)
            #     precision = precision_score(y, pred_oob1)
            #     print("precision", precision)
            #     accuracy = accuracy_score(y, pred_oob1)
            #     print("accuracy", accuracy)
            #     f1 = f1_score(y, pred_oob1)
            #     print("f1", f1)
            # except:
            #     pass
