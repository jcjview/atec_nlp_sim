import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
input_file="../input/process.csv"
df = pd.read_csv(input_file,encoding="utf-8")
print('Total number of question pairs for training: {}'.format(len(df)))

qids = pd.Series(df['question1'].tolist() + df['question2'].tolist())
print('Total number of questions in the training data: {}'.format(len(
    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

print('equal intent pairs: {}%'.format(round(df['label'].mean()*100, 2)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
plt.show()