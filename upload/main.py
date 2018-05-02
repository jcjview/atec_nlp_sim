#/usr/bin/env python
#coding=utf-8
import sys

import jieba

from upload.keras_model import keras_model

jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('余额宝')

def seg(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)



model=keras_model()

def process(inpath, outpath):
    test_data1 = []
    test_data2 = []
    linenos=[]
    with open(inpath, 'r') as fin:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            test_data1.append(seg(sen1))
            test_data2.append(seg(sen2))
            linenos.append(lineno)
    labels=model.predict(test_data1,test_data2)
    with open(outpath, 'w') as fout:
        for index,la in  enumerate(labels):
            lineno=linenos[index]
            fout.write(lineno + '\t%d\n'%la)

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
