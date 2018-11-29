import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from sklearn import svm, linear_model
from utils import untuple, retuple

#yf=[]
with open('y_train.txt', 'rb') as f:
    yf=pickle.load(f)
    f.close()

with open('X_train1.txt', 'rb') as f:
    X_train1=pickle.load(f)
    f.close()

with open('X_train2.txt', 'rb') as f:
    X_train2=pickle.load(f)
    f.close()

with open('X_train3.txt', 'rb') as f:
    X_train3=pickle.load(f)
    f.close()


untupyf=[]
for tup in yf:
    untupyf.append(untuple(tup))
labenc=LabelEncoder()
y_train=labenc.fit_transform(untupyf)

enc=OneHotEncoder(sparse=False)
enc.fit(X_train1)
X_train1Enc=enc.transform(X_train1)

clf = linear_model.Perceptron(max_iter=40, n_jobs=-1)
clf.fit(X_train1Enc, y_train)
print(clf.score(X_train1Enc, y_train))

