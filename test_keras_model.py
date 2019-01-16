import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pymysql.cursors
from keras.utils import to_categorical
model = keras.models.load_model('keras_example_with_modified_data_test.h5')

#get testing data
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
        sql = "select * from newspaper where label is not null and id > 600 limit 100"
        cursor.execute(sql)
        results = cursor.fetchall()
        for result in results:
            if(int(result.get('label'))>0):
                texts.append(result.get('content'))
                label_id = result.get('label')
                # label_id = len(labels_index)
                # labels_index[result.get('id')] = label_id
                labels.append(int(label_id)-1)
        # sql_label = "select label from newspaper where label is not null"
        # cursor.execute(sql_label)
        # label_results = cursor.fetchall()
        # for label_result in label_results:
        #     labels_index[int(label_result.get('label'))-1]=label_result.get('label')
    connection.commit()
finally:
    connection.close()
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
labels = to_categorical(np.asarray(labels))
data = data[indices]
labels = labels[indices]

print(model.evaluate(data,labels))
print( model.predict(data))