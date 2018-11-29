def extract_features(conf, original_sentence): #donne un onze-uet : on prend les features f3 + S.-1.POS

    l= []
    l.append(conf.stack[-1].getFeat('POS')) #0 S.0.POS
    l.append(conf.stack[-1].getFeat('LEMMA')) #1 S.0.LEMMA
    l.append(conf.stack[-1].getFeat('MORPHO')) #2 S.0.MORPHO



    l.append(conf.buffer[0].getFeat('POS'))#3 B.0.POS
    l.append(conf.buffer[0].getFeat('LEMMA')) #4 B.0.LEMMA
    l.append(conf.buffer[0].getFeat('MORPHO')) #5 B.0.MORPHO
    #print(len(original_sentence))
    if int(conf.buffer[0].getFeat('INDEX'))>0:
        l.append(original_sentence[int(conf.buffer[0].getFeat('INDEX'))-1].getFeat('POS')) #6 B.-1.POS
    else :
        l.append('empty')
    #print(int(conf.buffer[0].getFeat('INDEX')))
    if int(conf.buffer[0].getFeat('INDEX'))<(len(original_sentence)-1):
        l.append(original_sentence[int(conf.buffer[0].getFeat('INDEX')) + 1].getFeat('POS')) #7 B.1.POS
    else :
        l.append('empty')


    l.append(int(conf.buffer[0].getFeat('INDEX')) - int(conf.stack[-1].getFeat('INDEX'))) #8 DIST

    if len(conf.stack)>1:
        l.append(conf.stack[-2].getFeat('POS')) #9 S.1.POS
    else:
        l.append('empty')

    #l.append(conf.stack[-2].getFeat('POS')) #10 S.-1.POS

    return l

def select_trainfeatures(X, num):
    if num == 1:
        X_train = [X[0], X[3], X[8]]
    elif num ==2:
        X_train = X[:9]
    elif num ==3:
        X_train = X
    else :
        X_train=None
        print('invalid features map')
    return(X_train)


def cond_reduce(sig,s,conf): #true si toutes les dépendances de sig ont été faites
    res= True
    index = sig.getFeat('INDEX')
    for word in s:
        if word.getFeat('GOV') == index : #word est alors un dépendant de sig
            if (sig, word.getFeat('LABEL'), word) in conf.dependencies:
                pass
                print('pass')
            else :
                res= False
    return res

            #je regarde si conf.dependencies contient la dep (sig, word.getFeat('LABEL'), word)

def untuple(tupTransi):
    return tupTransi[0]+tupTransi[1]

def retuple(strTransi):
    return (strTransi[0][:-2], strTransi[0][-2:])

