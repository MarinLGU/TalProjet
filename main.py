print("Hello World")
from utils import non_projective
from WordBuffer import WordBuffer
from Configuration import Configuration
from AE import *
import pickle
import copy

train=True
inference=False
save= True
featset =1
langue='Dutch'
mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
#filename='UD_French-GSD/fr_gsd-ud-dev.conllu'

wb = WordBuffer(mcd)
if train:
    wb.readFromConlluFile('UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu')
else:
    wb.readFromConlluFile('UD_Dutch-LassySmall/nl_lassysmall-ud-test.conllu')
Xf = [] #de dimensions nb_config_extraites * 11
Yf = [] #de dimensions (2|L|+2) * 2
Zf = []


#etat_ini = next_sentence
#s = etat_ini.buffer
i=1
np=0
while wb.getCurrentIndex()<wb.getLength():
#while i<2:
    i+=1
    #print(wb.getCurrentIndex(), wb.getLength())
    conf=Configuration(wb)
    original_sentence=copy.copy(conf.buffer)
    if non_projective(original_sentence):
        np+=1
        print('non projective', i)
        continue
    else:
        X,Y, Z=arc_eager(conf,original_sentence, inference)
        Xf.extend(X), Yf.extend(Y), Zf.extend(Z)
    #print(len(Zf))

if not inference:
    Xf1=[]
    Xf2=[]
    Xf3=[]
    for features in Xf:
        #print(len(features))
        Xf1.append(select_trainfeatures(features, 1))
        Xf2.append(select_trainfeatures(features, 2))
        Xf3.append(select_trainfeatures(features,3))


print('nombre de phrases non projectives : ', np)
if save and train and not inference:


    with open('y_trainDutch.txt', 'wb') as f:
        pickle.dump(Yf, f)
        f.close()
    if featset == 1:
        with open('f1_utils/X_train1%s.txt'%langue, 'wb') as f:
            pickle.dump(Xf1, f)
            f.close()
    if featset==2:
        with open('f2_utils/X_train2%s.txt'%langue, 'wb') as f:
            #for item in Xf2:
                #f.write("%s\n" %str(item))
            pickle.dump(Xf2, f)
            f.close()
    if featset==3:
        with open('f3_utils/X_train3%s.txt'%langue, 'wb') as f:
            #for item in Xf3:
             #   f.write("%s\n" %str(item))
            pickle.dump(Xf3, f)
            f.close()


if save and inference and train:
    with open('dep_train_%i.txt'%featset, 'wb') as f:
        pickle.dump(Zf, f)
        f.close()

if save and not train and not inference:
    with open('y_test%s.txt'%langue, 'wb') as f:
        pickle.dump(Yf, f)
        f.close()


    if featset == 1:
        with open('f1_utils/X_test1%s.txt'%langue, 'wb') as f:
            print(len(Xf1))
            print('save Xf1 test')
            #for item in Xf1:
             #   f.write("%s\n" %str(item))
            pickle.dump(Xf1, f)
            f.close()
    if featset==2:
        with open('f2_utils/X_test2%s.txt'%langue, 'wb') as f:
            #for item in Xf2:
                #f.write("%s\n" %str(item))
            pickle.dump(Xf2, f)
            f.close()
    if featset==3:
        with open('f3_utils/X_test3%s.txt'%langue, 'wb') as f:
            #for item in Xf3:
             #   f.write("%s\n" %str(item))
            pickle.dump(Xf3, f)
            f.close()


