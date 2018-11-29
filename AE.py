from Transitions import *
from utils import *
import pickle
import numpy as np

with open('skperceptron.txt', 'rb') as f:
    clf=pickle.load(f)
    f.close()

with open('labenc.txt','rb') as f:
    labenc=pickle.load(f)
    f.close()

with open('Xenc1.txt', 'rb') as f:
    Xenc1=pickle.load(f)
    f.close()


def arc_eager(conf, original_sentence, inference=False):
    conf.stack.append(conf.buffer.pop(0))
    #conf.stack.append(conf.buffer.pop(0))  # initial shifts
    X=[]
    Y=[]

    if inference == False:
        while len(conf.buffer) != 0 and len(conf.stack) != 0:
            #print(len(conf.buffer), len(conf.stack) )

            x = extract_features(conf, original_sentence)
            X.append(x)

            beta, sig = conf.buffer[0], conf.stack[-1]
            #print(cond_reduce(sig, original_sentence, conf))

            if sig.getFeat('GOV') == beta.getFeat('INDEX'):
                transi = 'LA'
                y = (sig.getFeat('LABEL'), transi)

            elif beta.getFeat('GOV') == sig.getFeat('INDEX'):
                transi = 'RA'
                y = (beta.getFeat('LABEL'), transi)

            elif cond_reduce(sig,original_sentence,conf):
                transi = 'RE'
                y = ('', transi)
            else:
                transi = 'SH'
                y = ('', transi)
            Y.append(y)
            print(transi)
            transition(beta, sig, conf, transi)
    else:

        while len(conf.buffer) != 0 and len(conf.stack) != 0:
            X=select_trainfeatures(extract_features(conf, original_sentence), 1)
            #print(Xenc1.categories_)
            #X=[X[0],X[1], int(X[2])]
            #X=np.array(X).reshape(1,-1)
            X=[[X[0],X[1], int(X[2])]]
            print(X)
            X=Xenc1.transform(X)
            print(X)
            classe=clf.predict(X)
            etiq, transi = retuple(labenc.inverse_transform(classe))
            #l'étiquette ici n'importe pas mais le transi pris par transition
            beta, sig = conf.buffer[0], conf.stack[0]

            transition(beta, sig, conf, transi)
    return X, Y #X.shape=nbconfig*11 Y.shape=nbconfig*2