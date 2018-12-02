import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils import untuple, retuple
import numpy as np
save=True
langue='English'
with open('../y_train%s.txt'%langue, 'rb') as f:
    yf=pickle.load(f)
    f.close()

with open('X_train1%s.txt'%langue, 'rb') as f:
    X_train1=pickle.load(f)
    f.close()

with open('../y_test%s.txt'%langue, 'rb') as f:
    yftest=pickle.load(f)
    f.close()

with open('X_test1%s.txt'%langue, 'rb') as f:
    X_test1=pickle.load(f)
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

Xenc1=OneHotEncoder(sparse=False, handle_unknown='ignore')
Xenc1.fit(X_train1)
X_train1Enc=Xenc1.transform(X_train1)
X_test1Enc=Xenc1.transform(X_test1)

if save:
    with open('labenc%s.txt'%langue,'wb') as f:
        pickle.dump(labenc,f)
        f.close()

    with open('Xenc1%s.txt'%langue, 'wb') as f:
        pickle.dump(Xenc1, f)
        f.close()
    with open('X_train1Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(X_train1Enc, f)
        f.close()
    with open('X_test1Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(X_test1Enc, f)
        f.close()
    with open('y_train1Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(y_trainEnc, f)
        f.close()
    with open('y_test1Enc%s.txt'%langue, 'wb') as f:
        pickle.dump(y_testEnc, f)
        f.close()