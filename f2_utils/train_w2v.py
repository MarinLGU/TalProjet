from gensim.models import Word2Vec, FastText
from WordBuffer import WordBuffer
from Configuration import Configuration
import fasttext

mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
#filename='UD_French-GSD/fr_gsd-ud-dev.conllu'

wb = WordBuffer(mcd)
wb.readFromConlluFile('../UD_French-GSD/fr_gsd-ud-train.conllu')
wb2=WordBuffer(mcd)
wb2.readFromConlluFile('../UD_French-GSD/fr_gsd-ud-train.conllu')

lemmes=[]
while wb.getCurrentIndex()<wb.getLength():
    sentence=[]
    conf = Configuration(wb)
    for word in conf.buffer:
        sentence.append(word.getFeat('LEMMA'))
    lemmes.append(sentence)

while wb2.getCurrentIndex()<wb2.getLength():
    sentence=[]
    conf = Configuration(wb2)
    for word in conf.buffer:
        sentence.append(word.getFeat('LEMMA'))
    lemmes.append(sentence)

w2v=Word2Vec.load('../w2cpretrained/fr.bin')
w2v.build_vocab(lemmes, update=True)
w2v.train(lemmes, total_examples=len(lemmes), epochs=1)
w2v.save("w2vfrench.model")
