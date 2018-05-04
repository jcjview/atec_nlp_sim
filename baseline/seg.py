#py3
import sys
import jieba
import re
import numpy as np
jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('余额宝')

input_file="..\\data\\answers.txt"
# input_file="..\\input\\atec_nlp_sim_train.csv"
output_file="fc2.txt"
def seg(text):
    seg_list = jieba.cut(text.strip())
    return " ".join(seg_list)

# dict_file_name="../data/dict.txt"
# jieba.load_userdict(input_file)
# # jieba.add_word('花呗')
# df = pd.read_csv(input_file,encoding="utf-8")
# q=df["question1"]
# for s in q:
#     seg_list=jieba.cut(s)
#     print("/ ".join(seg_list))
#     break

# a=[0.1,0.5,0.8]
# l=[0,1,1]
# b=np.array(a)
# d =  (b>0.5).astype(int)
# # d=np.stack((b,c),axis=1)
# print(d)
# print(d.shape)
# from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
# s=f1_score(l,d)
# # b=d.argmax(axis=-1)
# print(s)
special_character_removal = re.compile(r'[@#$%^&*,.【】[]{}；‘，。、？!? \\/"\']', re.IGNORECASE)
replace_numbers = re.compile(r'\d+', re.IGNORECASE)
if __name__ == '__main__':

    with open(input_file,encoding="utf-8") as fp,open(output_file,"w",encoding="utf-8") as fw:
        for line in fp:
            line = special_character_removal.sub('', line)
            line = replace_numbers.sub('NUMBER_REPLACE', line)
            lines=line.strip().split(" ++$++ ")
            if(len(lines)==3):
                line=lines[1]
            fw.write(seg(line))
            fw.write("\n")


