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
texts = accessMysql.getContentList("select * from newspaper where label is not null and id <350")
labels = accessMysql.getLabelList("select * from newspaper where label is not null and id <350")
test_data = accessMysql.getContentList("select * from newspaper where label is not null and id >600")
test_labels = accessMysql.getLabelList("select * from newspaper where label is not null and id >600")
# print(texts)


tokens = vectorizeText.split_list(texts)
# print(tokens)
print('vectorize sentences')
model_link = '../models_bin/wiki.vi.model.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_link, binary=True)
# word2vec_model = Word2Vec.load(model_link)
vectors = [] 
for token in tokens:
    vectors.append(vectorizeText.sent_vectorize(token,word2vec_model))
# vector = np.asarray(vectors)
# print(vector.shape)
print('train model')

model = LogisticRegression(random_state=0,solver = 'lbfgs',multi_class='multinomial').fit(vectors,labels)

print("testing")
test_sequence_list = vectorizeText.split_list(test_data)
# print(test_sequence_list)
test_vectors = []
for test_sequence in test_sequence_list:
    test_vectors.append(vectorizeText.sent_vectorize(test_sequence,word2vec_model))
test_vectors = np.asarray(test_vectors)

test_result = model.predict(test_vectors)
print(accuracy_score(test_labels,test_result))
print(test_result)
print("--------------------------------------------------------------")
print(test_labels)

# tets_texts = accessMysql.getContentList("select * from newspaper where id = 1")
# words = vectorizeText.split_list(tets_texts)
# for word in words:
#     for single in word:
#         print(word2vec_model.word_vec(single))
