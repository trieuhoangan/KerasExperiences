import pymysql.cursors
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim import models
from gensim.models import KeyedVectors

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import keras.preprocessing.text as txtprc
# from customlib import vectorizeText
np.seterr(divide='ignore', invalid='ignore')
import sys
sys.path.append("..")
from customlib import vectorizeText,accessMysql
   
   

print("load data from mysql")
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

datas = []
for text in texts:
    datas.append(text.replace('_',' '))
test_cases = []
for test_case in test_data:
    test_cases.append(test_case.replace('_',' '))
tokens = vectorizeText.split_list(datas)

print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)

vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
vectors = np.nan_to_num(vectors)
print('train model')

model = LogisticRegression(random_state=0,solver = 'lbfgs',multi_class='multinomial').fit(vectors,labels)

print("testing")
test_sequence_list = vectorizeText.split_list(test_cases)
# print(test_sequence_list)
test_vectors = []
for test_sequence in test_sequence_list:
    test_vectors.append(vectorizeText.sent_vectorize(test_sequence,word2vec_model))
test_vectors = np.nan_to_num(test_vectors)

test_result = model.predict(test_vectors)
print(accuracy_score(test_labels,test_result))
print("test result:")
print(test_result)
print("--------------------------------------------------------------")
print("test examples:")
print(test_labels)

# tets_texts = accessMysql.getContentList("select * from newspaper where id = 1")
# words = vectorizeText.split_list(tets_texts)
# for word in words:
#     for single in word:
#         print(word2vec_model.word_vec(single))
