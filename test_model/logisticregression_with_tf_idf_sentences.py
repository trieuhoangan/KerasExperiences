import re
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


print("load data from mysql")
all_text = accessMysql.getContentList("select * from sentences")

sentence_label_0 = accessMysql.getContentList("select * from sentences where label = 0 limit 150")
label_type_0 = accessMysql.getLabelList("select * from sentences where label = 0  limit 150")
test_data_label_type_0 = accessMysql.getContentList("select * from sentences where label = 0  limit 150,50")
test_label_type_0 = accessMysql.getLabelList("select * from sentences where label = 0  limit 150,50")

sentence_label_1 = accessMysql.getContentList("select * from sentences where label = 1 limit 150")
label_type_1 = accessMysql.getLabelList("select * from sentences where label = 1  limit 150")
test_data_label_type_1 = accessMysql.getContentList("select * from sentences where label = 1  limit 150,50")
test_label_type_1 = accessMysql.getLabelList("select * from sentences where label = 1  limit 150,50")

sentence_label_2 = accessMysql.getContentList("select * from sentences where label = 2 limit 150")
label_type_2 = accessMysql.getLabelList("select * from sentences where label = 2  limit 150")
test_data_label_type_2 = accessMysql.getContentList("select * from sentences where label = 2  limit 150,50")
test_label_type_2 = accessMysql.getLabelList("select * from sentences where label = 2  limit 150,50")
texts = []
labels = []
test_data = []
test_labels = []
# print(label_type_0)
texts.extend(sentence_label_0)
texts.extend(sentence_label_1)
texts.extend(sentence_label_2)
labels.extend(label_type_0)
labels.extend(label_type_1)
labels.extend(label_type_2)
test_data.extend(test_data_label_type_0)
test_data.extend(test_data_label_type_1)
test_data.extend(test_data_label_type_2)
test_labels.extend(test_label_type_0)
test_labels.extend(test_label_type_1)
test_labels.extend(test_label_type_2)
print('preprocessing data')
valid_file_name_character = re.compile('[\\~#%&*{}/:<>?|\"-]')

words = []
for text in all_text:
    text = re.sub(valid_file_name_character,'',text)
    words.append(text.split(' '))
    
stop_word = ['.',',',';','!','@','#','-','>','(',')','/']
#create dictionary 

tf = TfidfVectorizer(min_df=15,max_df=0.7,sublinear_tf=True,encoding='utf-8',stop_words=stop_word,analyzer='word')
model = tf.fit(all_text)
vectors = tf.transform(texts)
print(vectors.shape)
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
print("test result:")
print(test_result)
print("--------------------------------------------------------------")
print("test examples:")
print(test_labels)