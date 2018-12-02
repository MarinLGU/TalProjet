import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils import untuple, retuple
import numpy as np
import fasttext

save=True
langue='English'
if langue=='Dutch':
    filename='../w2cpretrained/nl.bin'
if langue=='English':
    filename='../w2cpretrained/wiki-news-300d-1M.vec'
if langue=='French':
    filename = '../w2cpretrained/fr.bin'

with open('../y_train%s.txt'%langue, 'rb') as f:
    yf=pickle.load(f)
    f.close()

with open('X_train3%s.txt'%langue, 'rb') as f:
    X_train2=pickle.load(f)
    f.close()

with open('../y_test%s.txt'%langue, 'rb') as f:
    yftest=pickle.load(f)
    f.close()

with open('X_test3%s.txt'%langue, 'rb') as f:
    X_test2=pickle.load(f)
    f.close()


total=yf+yftest
untupyf=[]
untupyftest=[]
untuptotal=[]


for tup in total:
    untuptotal.append(untuple(tup))

for tup in yf:
    untupyf.append(untuple(tup))
for tup in yftest:
    untupyftest.append(untuple(tup))
labenc=LabelEncoder()
labenc.fit(untuptotal)


y_trainEnc=labenc.transform(untupyf)
y_testEnc=labenc.transform(untupyftest)

def encodeLemma(X):
    ftlemma1=[]
    ftlemma2=[]
    ft=fasttext.load_model(filename)
    for features in X:
        ftlemma1.append(ft[features.pop(1)])
        ftlemma2.append(ft[features.pop(2)])
    return ftlemma1,ftlemma2


ftTrainlemma1, ftTrainlemma2=encodeLemma(X_train2)
ftTestlemma1, ftTestlemma2=encodeLemma(X_test2)
X_train2=np.array(X_train2)
X_test2=np.array(X_test2)
total=np.concatenate([X_train2, X_test2])
vocs = [np.array(list(set(total[:,i]))) for i in range(np.shape(total)[1])]
X_train2Enc = np.array([[np.argwhere(vocs[j]==X_train2[i,j]) for j,feature in enumerate(features)] for i,features in enumerate(X_train2)]).reshape(len(X_train2),8)
X_test2Enc = np.array([[np.argwhere(vocs[j]==X_test2[i,j]) for j,feature in enumerate(features)] for i,features in enumerate(X_test2)]).reshape(len(X_test2), 8)

#Xenc2=OneHotEncoder(sparse=False, handle_unknown='ignore')
#Xenc2.fit(X_train2)
#X_train2Enc=Xenc2.transform(X_train2)
X_train2Enc=np.concatenate([X_train2Enc, ftTrainlemma1, ftTrainlemma2], axis=-1)
#X_test2Enc=Xenc2.transform(X_test2)
X_test2Enc=np.concatenate([X_test2Enc, ftTestlemma1, ftTestlemma2], axis=-1)

if save:
    with open('labenc%s.txt'%langue,'wb') as f:
        pickle.dump(labenc,f)
        f.close()

    # with open('Xenc2.txt', 'wb') as f:
    #     pickle.dump(Xenc1, f)
    #     f.close()
    with open('X_train3Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(X_train2Enc, f)
        f.close()
    with open('X_test3Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(X_test2Enc, f)
        f.close()
    with open('y_train3Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(y_trainEnc, f)
        f.close()
    with open('y_test3Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(y_testEnc, f)
        f.close()

