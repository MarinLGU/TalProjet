print("Hello World")
from Dicos import Dicos
from WordBuffer import WordBuffer

mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
filename='UD_French-GSD/fr_gsd-ud-dev.conllu'
dc=Dicos(mcd=mcd, verbose=True)
dc.populateFromConlluFile(filename, verbose=True)

wb = WordBuffer(mcd)
wb.readFromConlluFile('UD_French-GSD/fr_gsd-ud-train.conllu')
print(word for word in wb.)
