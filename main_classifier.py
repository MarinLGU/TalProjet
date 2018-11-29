import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from sklearn import svm, linear_model
from utils import untuple, retuple

save=True
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

Xenc1=OneHotEncoder(sparse=False, handle_unknown='ignore')
Xenc1.fit(X_train1)
X_train1Enc=Xenc1.transform(X_train1)

clf = linear_model.Perceptron(max_iter=5, n_jobs=-1)
clf.fit(X_train1Enc, y_train)
print(clf.score(X_train1Enc, y_train))

if save:
    with open('skperceptron.txt', 'wb') as f:
        pickle.dump(clf, f)
        f.close()

    with open('labenc.txt','wb') as f:
        pickle.dump(labenc,f)
        f.close()

    with open('Xenc1.txt', 'wb') as f:
        pickle.dump(Xenc1, f)
        f.close()



