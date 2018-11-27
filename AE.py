from Transitions import *
from utils import *
import numpy as np
def arc_eager(conf, s, inference=False):
    conf.stack.append(conf.buffer.pop(0))
    #conf.stack.append(conf.buffer.pop(0))  # initial shifts
    X=[]
    Y=[]

    if inference == False:
        while len(conf.buffer) != 0 and len(conf.stack) != 0:
            x = extract_features(conf)
            X.append(x)

            beta, sig = conf.buffer[0], conf.stack[-1]

            if sig.getFeat('GOV') == beta.getFeat('INDEX'):
                transi = 'LA'
                y = (sig.getFeat('LABEL'), transi)

            if beta.getFeat('GOV') == sig.getFeat('INDEX'):
                transi = 'RA'
                y = (beta.getFeat('LABEL'), transi)

            if cond_reduce(sig,s,conf):
                transi = 'RE'
                y = ('', transi)
            else:
                transi = 'SH'
                y = ('', transi)
            Y.append(y)
            transition(beta, sig, conf, transi)
    else:

        while len(conf.buffer) != 0 and len(conf.stack) != 0:
            etiq, transi = clf.predict(extract_features(conf))

            beta, sig = conf.buffer[0], conf.stack[0]

            transition(beta, sig, conf, transi)
    return X, Y #X.shape=nbconfig*11 Y.shape=nbconfig*2