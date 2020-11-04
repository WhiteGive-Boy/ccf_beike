import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.callbacks import Callback
train_left = pd.read_csv('./train/train.query.tsv',sep='\t',header=None)
train_left.columns=['id','q1']
train_right = pd.read_csv('./train/train.reply.tsv',sep='\t',header=None)
train_right.columns=['id','id_sub','q2','label']
df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('好的')
test_left = pd.read_csv('./test/test.query.tsv',sep='\t',header=None, encoding='gbk')
test_left.columns = ['id','q1']
test_right =  pd.read_csv('./test/test.reply.tsv',sep='\t',header=None, encoding='gbk')
test_right.columns=['id','id_sub','q2']
df_test = test_left.merge(test_right, how='left')
PATH = './'
BERT_PATH = './'
WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 100
input_categories = ['q1','q2']
output_categories = 'label'

maxlen = 100
learning_rate = 5e-5
min_learning_rate = 1e-5
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            np.random.shuffle(list(idxs))
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                text2 = d[1][:maxlen]
                x1, x2 = tokenizer.encode(first=text,second=text2)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


tokenizer = OurTokenizer(token_dict)

data=df_train[['q1','q2','label']].to_numpy()
random_order = range(len(data))
np.random.shuffle(list(random_order))
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]



bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()




train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
testdata=df_test[['q1','q2']].to_numpy()
def makeresult(testdata):
    result=[]
    for test in testdata:
        _t1, _t2 = tokenizer.encode(first=test[0],second=test[1])
        _t1, _t2 = np.array([_t1]), np.array([_t2])
        label = model.predict([_t1, _t2])
        result.append([label])
    return result
result=makeresult(testdata)
df_test['label']=result
df_test=df_test[['id','id_sub','label']]
df_test.to_csv("result.csv",index=0)

result = pd.read_csv('./result.csv')
result['newlabel']=result['label'].apply(lambda x:re.findall(u'.*\\[\\[(.*)\\]\\].*', x))
result['newlabel']=result['newlabel'].apply(lambda x:x[0])
result['newlabel']=result['newlabel'].apply(lambda x:1 if float(x)>=0.5 else 0)
result=result[['id','id_sub','newlabel']]
# print(result['newlabel'])
result.to_csv("newresult.tsv",sep='\t',header=None,index=0)




