import pickle
from utils import untuple
with open('y_test.txt', 'rb') as f:
    y_test=pickle.load(f)
    f.close()

with open('f1_utils/X_test1.txt', 'rb') as f:
    X_test1=pickle.load(f)
    f.close()


with open('skperceptron.txt', 'rb') as f:
    clf=pickle.load(f)
    f.close()

with open('labenc.txt','rb') as f:
    labenc=pickle.load(f)
    f.close()

with open('f1_utils/Xenc1.txt', 'rb') as f:
    Xenc1=pickle.load(f)
    f.close()

untupYtest=[]
for tup in y_test:
    untupYtest.append(untuple(tup))
untupYtest=labenc.transform(untupYtest)

X_test1Enc=Xenc1.transform(X_test1)
#print(X_test1Enc.shape)
print(clf.score(X_test1Enc, untupYtest))
