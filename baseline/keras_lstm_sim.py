import pandas as pd
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score,recall_score,precision_score

MAX_TEXT_LENGTH=50
MAX_FEATURES=72039
embedding_dims=100
dr = 0.2

def get_model():
    input1_tensor = keras.layers .Input(shape=(MAX_TEXT_LENGTH,))
    input2_tensor = keras.layers.Input(shape=(MAX_TEXT_LENGTH,))
    words_embedding_layer = keras.layers.Embedding(MAX_FEATURES, embedding_dims,trainable=True)
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

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['f1_score'] = float('0')
            logs['recall'] = float('0')
            if (self.validation_data):
                print(self.validation_data.shape)
                print(self.validation_data[0].shape)
                print(self.validation_data[1].shape)
                predict = self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)
                logs['recall'] = f1_score(self.validation_data[1], predict)
                logs['f1_score'] = recall_score(self.validation_data[1], predict)
    def on_train_begin(self, logs={}):
        if not ('f1_score' in self.params['metrics']):
            self.params['metrics'].append('f1_score')
        if not ('recall' in self.params['metrics']):
            self.params['metrics'].append('recall')

    def on_epoch_end(self, epoch, logs={}):
        logs['f1_score'] = float('0')
        logs['recall']=float('0')
        if (self.validation_data):
            print(self.validation_data.shape)
            print(self.validation_data[0].shape)
            print(self.validation_data[1].shape)
            predict=self.model.predict(self.validation_data[0], batch_size=self.predict_batch_size)
            logs['recall'] = f1_score(self.validation_data[1], predict)
            logs['f1_score'] = recall_score(self.validation_data[1],predict)


input_file="../input/process.csv"
df = pd.read_csv(input_file,encoding="utf-8")

question1=df['question1'].values
question2=df['question2'].values
y = df['label'].values
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(question1)+list(question2))
list_tokenized_question1 = tokenizer.texts_to_sequences(question1)
list_tokenized_question2 = tokenizer.texts_to_sequences(question2)
X_train_q1 = pad_sequences(list_tokenized_question1, maxlen=MAX_TEXT_LENGTH)
X_train_q2 = pad_sequences(list_tokenized_question2, maxlen=MAX_TEXT_LENGTH)
seed=20180426
cv_folds=10
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=cv_folds, random_state=seed, shuffle=False)
pred_oob = np.zeros(y.shape)
for ind_tr, ind_te in skf.split(X_train_q1, y):
    x_train_q1 = X_train_q1[ind_tr]
    x_train_q2 = X_train_q2[ind_tr]
    x_val_q1 = X_train_q1[ind_te]
    x_val_q2 = X_train_q2[ind_te]
    y_train=y[ind_tr]
    y_val=y[ind_te]
    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    bst_model_path =  'weight.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', mode='min',
                                       save_best_only=True, verbose=1, save_weights_only=True)
    hist = model.fit([x_train_q1,x_train_q2], y_train,
                     validation_data=([x_val_q1,x_val_q2], y_val),
                     epochs=50, batch_size=256, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint,F1ScoreCallback()])
    y_predict=model.predict([x_val_q1,x_val_q2])
    pred_oob[ind_te]=y_predict
recall=recall_score(y,pred_oob)
print("recal",recall)
precision=precision_score(y,y_predict)
print("precision",precision)
f1=f1_score(y,y_predict)
print("f1",f1)