from Transitions import *
from utils import *
import pickle
import numpy as np
from Word import Word
mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))

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
        while len(conf.buffer) != 0 or len(conf.stack) != 0:
            #print(len(conf.buffer), len(conf.stack) )

            x = extract_features(conf, original_sentence)
            X.append(x)
            if len(conf.buffer) == 0:
                conf.buffer.append(Word.emptyWord(mcd))
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

        while len(conf.buffer) != 0 or len(conf.stack) != 0:
            print(extract_features(conf, original_sentence))
            x=select_trainfeatures(extract_features(conf, original_sentence), 1)
            X.append(x)
            #print(Xenc1.categories_)
            #X=[X[0],X[1], int(X[2])]
            #X=np.array(X).reshape(1,-1)
            x=[[x[0],x[1], int(x[2])]]
            #print(X)
            x=Xenc1.transform(x)
            #print(X)
            classe=clf.predict(x)
            etiq, transi = retuple(labenc.inverse_transform(classe))
            #l'étiquette ici n'importe pas mais le transi pris par transition
            beta, sig = conf.buffer[0], conf.stack[0]
            Y.append((etiq, transi))
            print(transi)

            transition(beta, sig, conf, transi)

    print(len(conf.buffer),len(conf.stack) )
    return X, Y #X.shape=nbconfig*11 Y.shape=nbconfig*2