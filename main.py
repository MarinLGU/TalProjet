print("Hello World")
from utils import select_trainfeatures
from WordBuffer import WordBuffer
from Configuration import Configuration
from AE import *
import copy
inference=False
mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
filename='UD_French-GSD/fr_gsd-ud-dev.conllu'

wb = WordBuffer(mcd)
wb.readFromConlluFile('UD_French-GSD/fr_gsd-ud-train.conllu')
Xf = [] #de dimensions nb_config_extraites * 11
Yf = [] #de dimensions (2|L|+2) * 2


#etat_ini = next_sentence
#s = etat_ini.buffer

while wb.getCurrentIndex()<wb.getLength():
    conf=Configuration(wb)
    original_sentence=copy.copy(conf.buffer)
    X,Y=arc_eager(conf,original_sentence, inference)
    Xf.extend(X), Yf.extend(Y)

Xf1=[]
Xf2=[]
Xf3=[]
for features in Xf:
    #print(len(features))
    Xf1.append(select_trainfeatures(features, 1))
    Xf2.append(select_trainfeatures(features, 2))
    Xf3.append(select_trainfeatures(features,3))




with open('y_train.txt', 'w') as f:
    for item in Yf:
        f.write("%s\n" %str(item))
    f.close()
with open('X_train1.txt', 'w') as f:
    for item in Xf1:
        f.write("%s\n" %str(item))
    f.close()

with open('X_train2.txt', 'w') as f:
    for item in Xf2:
        f.write("%s\n" %str(item))
    f.close()

with open('X_train3.txt', 'w') as f:
    for item in Xf3:
        f.write("%s\n" %str(item))
    f.close()


