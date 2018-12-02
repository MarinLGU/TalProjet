import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from sklearn import svm, linear_model
from utils import untuple, retuple
import keras_simple_perceptron
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

with open('y_train.txt', 'rb') as f:
    yf=pickle.load(f)
    f.close()



with open('X_train2.txt', 'rb') as f:
    X_train2=pickle.load(f)
    f.close()




