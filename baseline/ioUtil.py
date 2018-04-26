import pandas as pd
import pandas as pd
import jieba
input_file='../input/atec_nlp_sim_train.csv'
ret=[]
jieba.add_word('花呗')
jieba.add_word('借呗')
jieba.add_word('余额宝')
jieba.add_word('***')

def seg(text):
    seg_list = jieba.cut(text)
    return " ".join(seg_list)

with open(input_file,encoding="utf-8") as fp:
    for line in fp:
        q={}
        lines=line.split("\t")
        if(len(lines)==3):
            q['question1']=seg(lines[0].strip())
            q['question2']=seg(lines[1].strip())
            q['label']=lines[2].strip()
        else:
            print(line)
        ret.append(q)
    df = pd.DataFrame(ret)
    df.to_csv("../input/process.csv",encoding="utf-8",index=False)

