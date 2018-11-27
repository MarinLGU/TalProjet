print("Hello World")
from Dicos import Dicos
from WordBuffer import WordBuffer
from Configuration import Configuration
from AE import *

mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
filename='UD_French-GSD/fr_gsd-ud-dev.conllu'
dc=Dicos(mcd=mcd, verbose=False)
dc.populateFromConlluFile(filename, verbose=False)

wb = WordBuffer(mcd)
wb.readFromConlluFile('UD_French-GSD/fr_gsd-ud-train.conllu')

cf=Configuration(wb)

Xf = [] #de dimensions nb_config_extraites * 11
Yf = [] #de dimensions (2|L|+2) * 2


#etat_ini = next_sentence
#s = etat_ini.buffer

while wb.getCurrentIndex()<wb.getLength():
    conf=Configuration(wb)
    s=conf.buffer
    X,Y=arc_eager(conf,s)
    Xf.extend(X), Yf.extend(Y)



