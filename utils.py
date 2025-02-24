def extract_features(conf, original_sentence): #donne un onze-uet : on prend les features f3 + S.-1.POS

    l= []
    l.append(conf.stack[-1].getFeat('POS')) #0 S.0.POS
    l.append(conf.stack[-1].getFeat('LEMMA')) #1 S.0.LEMMA
    l.append(conf.stack[-1].getFeat('MORPHO')) #2 S.0.MORPHO


    if len(conf.buffer)>0:
        l.append(conf.buffer[0].getFeat('POS'))#3 B.0.POS
    else:
        l.append('empty')
    if len(conf.buffer) > 0:
        l.append(conf.buffer[0].getFeat('LEMMA')) #4 B.0.LEMMA
    else:
        l.append('empty')
    if len(conf.buffer) > 0:
        l.append(conf.buffer[0].getFeat('MORPHO')) #5 B.0.MORPHO
    else:
        l.append('empty')
    if len(conf.buffer) > 0:
        if int(conf.buffer[0].getFeat('INDEX'))>0:
            l.append(original_sentence[int(conf.buffer[0].getFeat('INDEX'))-1].getFeat('POS')) #6 B.-1.POS
        else :
            l.append('empty')
    else :
        l.append('empty')
    if len(conf.buffer) > 0:
        if int(conf.buffer[0].getFeat('INDEX'))<(len(original_sentence)-1):
            l.append(original_sentence[int(conf.buffer[0].getFeat('INDEX')) + 1].getFeat('POS')) #7 B.1.POS
        else :
            l.append('empty')
    else:
        l.append('empty')

    if len(conf.buffer) > 0:
        l.append(int(conf.buffer[0].getFeat('INDEX')) - int(conf.stack[-1].getFeat('INDEX'))) #8 DIST
    else:
        l.append(0)

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
        if sig.getFeat('GOV')==word.getFeat('INDEX'):
            if (word, sig.getFeat('LABEL'), sig) in conf.dependencies:
                pass
            else:
                res=False
        if word.getFeat('GOV') == index : #word est alors un dépendant de sig
            if (sig, word.getFeat('LABEL'), word)  in conf.dependencies:
                pass
            else :
                res= False
    return res

#je regarde si conf.dependencies contient la dep (sig, word.getFeat('LABEL'), word)
def singheadcheck(sig,conf): #vérifie que sig n'a pas de gouverneur
    res = True
    for (w1,r,w2) in conf.dependencies:
        if w2==sig:
            res=False
        else:
            pass
    return res


def untuple(tupTransi):
    return tupTransi[0]+tupTransi[1]

def retuple(strTransi):
    return (strTransi[0][:-2], strTransi[0][-2:])

def non_projective(sentence):
    for word1 in sentence:
        if word1.getFeat('GOV')=='ROOT':
            dep1=0
        else:
            dep1=int(word1.getFeat('GOV'))
        if word1.getFeat('GOV') == 'ROOT':
            head1=0
        else:
            head1 = int(word1.getFeat('INDEX'))
        for word2 in sentence:
            if word2.getFeat('GOV') == 'ROOT':
                dep2 = 0
            else:
                dep2 = int(word2.getFeat('GOV'))
            if word2.getFeat('GOV') == 'ROOT':
                head2 = 0
            else:
                head2 = int(word2.getFeat('INDEX'))
            if head1 < 0 or head2 < 0:
                continue
            if (dep1 > head2 and dep1 < dep2 and head1 < head2) or (dep1 < head2 and dep1 > dep2 and head1 < dep2):
                return True

            if dep1 < head1 and head1 is not head2:
                if (head1 > head2 and head1 < dep2 and dep1 < head2) or (
                        head1 < head2 and head1 > dep2 and dep1 < dep2):
                    return True
    return False

def write_conllu(sentences, filename_in, filename_out):
#     features = ['INDEX', 'FORM', 'LEMMA', 'POS', 'X1', 'MORPHO', 'GOV', 'LABEL', 'X2', 'X3']
    with open(filename_in, 'r', encoding='utf-8') as fin, open(filename_out, 'w', encoding='utf-8') as fout:
        sentence_i = 0
        sentence_begin = True
        for line in fin:
            if line[0] == '\n':
                fout.write(line)
                if not sentence_begin:
                    sentence_i += 1
                    sentence_begin = True
            elif line[0] != '#':
                sentence_begin = False
                tokens = line.split('\t')
                if '-' in tokens[0]:
                    fout.write(line)
                    continue
                if int(tokens[0]) - 1 >= len(sentences[sentence_i]):
                    print(line)
                    print(sentences[sentence_i])
                    print(sentence_i)
                    break
                if sentences[sentence_i][int(tokens[0])].getFeat('GOV') =='nonprojective':
                    continue
                tokens[6] = sentences[sentence_i][int(tokens[0])].getFeat('GOV')
                tokens[7] = sentences[sentence_i][int(tokens[0])].getFeat('LABEL')
                fout.write('\t'.join(tokens))

def isallowed(conf, sig, transi):
    if transi=='LA':
        if not singheadcheck(sig, conf) and sig.getFeat('INDEX')!=0:
            return False
    if transi=='SH':
        if len(conf.buffer)>0:
            return False
    else:
        return True


