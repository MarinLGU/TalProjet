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
        i=0
        while len(conf.buffer) != 0 or len(conf.stack) != 0:
            i+=1
            # for dep in conf.dependencies:
            #     w1, tp, w2 = dep
            #     print(w1.getFeat('FORM'), tp, w2.getFeat('FORM'))
            # print(len(conf.stack), len(conf.buffer) )

            x = extract_features(conf, original_sentence)
            X.append(x)
            if len(conf.buffer) == 0:
                #conf.buffer.append(Word.emptyWord(mcd))
                beta=Word.emptyWord(mcd)
                sig=conf.stack[-1]
            else:
                beta, sig = conf.buffer[0], conf.stack[-1]

            #print('stack-1', sig.getFeat('FORM'), 'buf0', beta.getFeat('FORM'))

            if sig.getFeat('GOV') == beta.getFeat('INDEX') and singheadcheck(sig, conf) and sig.getFeat('INDEX')!=0:
                transi = 'LA'
                y = (sig.getFeat('LABEL'), transi)

            elif beta.getFeat('GOV') == sig.getFeat('INDEX') :
                transi = 'RA'
                y = (beta.getFeat('LABEL'), transi)

            elif cond_reduce(sig,original_sentence,conf):
                transi = 'RE'
                y = ('', transi)

            elif len(conf.buffer) >0:
                transi = 'SH'
                y = ('', transi)
            else:
                continue
            Y.append(y)
            transition(beta, sig, conf, transi)
    else:

        while len(conf.buffer) != 0 or len(conf.stack) != 0:
            #print(extract_features(conf, original_sentence))
            x=select_trainfeatures(extract_features(conf, original_sentence), 1)
            #x=extract_features(conf, original_sentence)
            X.append(x)
            x=[[x[0],x[1], int(x[2])]]
            #x=[[x[0], x[3], int(x[8])]]
            x=Xenc1.transform(x)
            #print(X)
            classe=clf.predict(x)
            etiq, transi = retuple(labenc.inverse_transform(classe))
            #l'étiquette ici n'importe pas mais le transi pris par transition
            if len(conf.buffer) == 0:
                #conf.buffer.append(Word.emptyWord(mcd))
                beta=Word.emptyWord(mcd)
                sig=conf.stack[-1]
            else:
                beta, sig = conf.buffer[0], conf.stack[-1]
            Y.append((etiq, transi))
            #print(transi)

            transition(beta, sig, conf, transi)

    #print(len(conf.buffer),len(conf.stack) )
    return X, Y #X.shape=nbconfig*11 Y.shape=nbconfig*2