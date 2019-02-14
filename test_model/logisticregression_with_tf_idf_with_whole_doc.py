import re
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

print("load data from mysql")
all_text = accessMysql.getContentList("select * from newspaper")
texts = accessMysql.getContentList("select * from newspaper where label is not null and id <350")
labels = accessMysql.getLabelList("select * from newspaper where label is not null and id <350")
test_data = accessMysql.getContentList("select * from newspaper where label is not null and id >600")
test_labels = accessMysql.getLabelList("select * from newspaper where label is not null and id >600")

print('preprocessing data')
valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')
sentences = []
words = []
for text in all_text:
    text = re.sub(valid_file_name_character,'',text)
    words.append(text.split(' '))
    group_sentences = text.split('.')
    for sentence in group_sentences:
        sentences.append(sentence)

stop_word = ['.',',',';','!','@','#','-','>','(',')','/']
#create dictionary 

tf = TfidfVectorizer(min_df=8,max_df=0.75,sublinear_tf=True,encoding='utf-8',stop_words=stop_word,analyzer='word')
model = tf.fit(sentences)
vectors = tf.transform(texts)

# for text in texts:
#     text = re.sub(valid_file_name_character,'',text)
#     vectors.append(tf.transform(text.split('.')))
#     # print(vectors)
print('training model ')
# print(vectors.shape)
LogisticRegressionModel = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(vectors, labels)
print('testing')
test_vectors = tf.transform(test_data)
test_result = LogisticRegressionModel.predict(test_vectors)
print(accuracy_score(test_labels,test_result))
print(test_result)
print("--------------------------------------------------------------")
print(test_labels)