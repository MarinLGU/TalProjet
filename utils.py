def extract_features(conf): #donne un onze-uet : on prend les features f3 + S.-1.POS

    l= []
    l.append(conf.stack[-1].getFeat('POS')) #0 S.0.POS
  #  l.append(conf.stack[-1].getFeat('LEMMA')) #1 S.0.LEMMA
   # l.append(conf.stack[-1].getFeat('MORPHO')) #2 S.0.MORPHO
#    l.append(conf.stack[-2].getFeat('POS')) #3 S.1.POS
    l.append(conf.buffer[0].getFeat('POS'))#4 B.0.POS
    #l.append(conf.buffer[0].getFeat('LEMMA')) #5 B.0.LEMMA
    #l.append(conf.buffer[0].getFeat('MORPHO')) #6 B.0.MORPHO
    #l.append(conf.buffer[1].getFeat('POS')) #7 B.-1.POS
    #l.append(conf.buffer[1].getFeat('POS')) #8 B.1.POS
#    l.append(int(conf.buffer[0].getFeat('INDEX')) - int(conf.stack[-1].getFeat('INDEX'))) #9 DIST

    l.append(conf.stack[-1].getFeat('POS')) #10 S.-1.POS

    return(l)

def select_trainfeatures(X, num):
    if num == 1:
        X_train = [X[0], X[4], X[9]]
    if num ==2:
        X_train = X[:10]
    if num ==3:
        X_train = X[0:3]+X[4:]
    return(X_train)


def cond_reduce(sig,s, conf): #true si toutes les dépendances de sig ont été faites
    res= True
    index = sig.getFeat('INDEX')
    for word in s:
        if word.getFeat('GOV') == index : #word est alors un dépendant de sig
            if (sig, word.getFeat('LABEL'), word) in conf.dependencies:
                pass
            else :
                res= False
    return res

            #je regarde si conf.dependencies contient la dep (sig, word.getFeat('LABEL'), word)
