'''This script loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import pymysql.cursors
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# from sklearn.svm import SVC

BASE_DIR = ''
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2



# prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
connection = pymysql.connect(host='localhost',
                            user='root',
                            password='12345678',
                            db='AliceII',
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        sql = "select * from newspaper where label is not null and id < 500"
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            if(int(result.get('label'))>0):
                texts.append(result.get('content'))
                label_id = result.get('label')
                # label_id = len(labels_index)
                # labels_index[result.get('id')] = label_id
                labels.append(int(label_id)-1)
        sql_label = "select label from newspaper where label is not null group by label order by label ASC"
        cursor.execute(sql_label)
        label_results = cursor.fetchall()
        label_count = 0
        while(label_count<5):
            labels_index[label_count]=label_results[label_count].get('label')
            label_count = label_count+1
        # for label_result in label_results:
        #     labels_index[int(label_result.get('label'))-1]=label_result.get('label')
    connection.commit()
finally:
    connection.close()

print(labels_index)
print('---------------------------------------------------------------------------------')    
print(labels)

# finally, vectorize the text samples into a 2D integer tensor
#---------------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
#---------------------------------------------------------------------
# split the data into a training set and a validation set
#---------------------------------------------------------------------
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
#---------------------------------------------------------------------
model = 'wiki.vi.model.bin'
from gensim.models import KeyedVectors
word2vec_model = KeyedVectors.load_word2vec_format(model, binary=True)
embedding_layer = word2vec_model.get_keras_embedding()
print('Training model.')
#---------------------------------------------------------------------
# train a 1D convnet with global maxpooling

#---------------------------------------------------------------------
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(300, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(300, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(300, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(300, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
#---------------------------------------------------------------------
model.fit(x_train, y_train,
          batch_size=128,
          epochs=5,
          validation_data=(x_val, y_val))

# model.fit(data, labels,
#           batch_size=128,
#           epochs=10,
#           validation_split=0.2)

#---------------------------------------------------------------------
#SVC
# model = SVC(kernel='linear',C=1.0,random_state=101)
# model.fit(data,labels) 
#---------------------------------------------------------------------

model.save('keras_example_with_modified_data_test.h5')