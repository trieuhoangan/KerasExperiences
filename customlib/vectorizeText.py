import numpy as np 
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
import pymysql.cursors
def sent_vectorize(sent, model):
    sent_vec = np.zeros(400)
    for single_sent in sent:
        try:
            sent_vec = np.add(sent_vec,model[single_sent])
        except:
            pass
    return sent_vec/np.sqrt(sent_vec.dot(sent_vec))

def split_list(doc_list):
    
    tokens = []
    for a in doc_list:
        tokens.append(a.split(' '))
    return tokens   
    