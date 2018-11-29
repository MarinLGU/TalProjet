from Transitions import *
from utils import *
from main_classifier import clf
import numpy as np
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
            etiq, transi = clf.predict(extract_features(conf))

            beta, sig = conf.buffer[0], conf.stack[0]

            transition(beta, sig, conf, transi)
    return X, Y #X.shape=nbconfig*11 Y.shape=nbconfig*2