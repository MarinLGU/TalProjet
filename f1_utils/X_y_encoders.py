import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils import untuple, retuple

save=True
with open('../y_train.txt', 'rb') as f:
    yf=pickle.load(f)
    f.close()

with open('X_train1.txt', 'rb') as f:
    X_train1=pickle.load(f)
    f.close()

with open('../y_test.txt', 'rb') as f:
    yftest=pickle.load(f)
    f.close()

with open('X_test1.txt', 'rb') as f:
    X_test1=pickle.load(f)
    f.close()



untupyf=[]
untupyftest=[]

for tup in yf:
    untupyf.append(untuple(tup))
for tup in yftest:
    untupyftest.append(untuple(tup))
labenc=LabelEncoder()
y_trainEnc=labenc.fit_transform(untupyf)
y_testEnc=labenc.transform(untupyftest)

Xenc1=OneHotEncoder(sparse=False, handle_unknown='ignore')
Xenc1.fit(X_train1)
X_train1Enc=Xenc1.transform(X_train1)
X_test1Enc=Xenc1.transform(X_test1)

if save:
    with open('labenc.txt','wb') as f:
        pickle.dump(labenc,f)
        f.close()

    with open('Xenc1.txt', 'wb') as f:
        pickle.dump(Xenc1, f)
        f.close()
    with open('X_train1Enc.txt', 'wb') as f:
        pickle.dump(X_train1Enc, f)
        f.close()
    with open('X_test1Enc.txt', 'wb') as f:
        pickle.dump(X_train1Enc, f)
        f.close()
    with open('y_train1Enc.txt', 'wb') as f:
        pickle.dump(y_trainEnc, f)
        f.close()
    with open('y_test1Enc.txt', 'wb') as f:
        pickle.dump(y_testEnc, f)
        f.close()