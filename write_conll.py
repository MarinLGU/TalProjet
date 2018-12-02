from WordBuffer import WordBuffer
from Configuration import Configuration
from AE import *
from utils import write_conllu, non_projective
import pickle
import copy


inference=True
save= False
mcd =(('INDEX', 'INT'), ('FORM', 'INT'), ('LEMMA', 'INT'), ('POS', 'SYM'), ('X1', 'INT'), ('MORPHO', 'INT'), ('GOV', 'SYM'), ('LABEL', 'SYM'), ('X2', 'SYM'), ('X3', 'SYM'))
filename='UD_French-GSD/fr_gsd-ud-dev.conllu'

wb_original = WordBuffer(mcd)
wb_original.readFromConlluFile('UD_French-GSD/fr_gsd-ud-train.conllu')
with open('dependencies.txt', 'rb') as f:
    dependencies=pickle.load(f)
    f.close()

i=0
sentences_label=[]
sentences_predicted=[]
while wb_original.getCurrentIndex()<wb_original.getLength():
#while i<3:

    #print(wb.getCurrentIndex(), wb.getLength())
    confo=Configuration(wb_original)
    confp=copy.deepcopy(confo)
    if non_projective(confo.buffer):
        for word in confo.buffer:
            word.setFeat('GOV', 'nonprojective')
            word.setFeat('LABEL', 'nonprojective')
        sentences_label.append(confp.buffer)
        sentences_predicted.append(confp.buffer)
    else:
        #for word in confo.buffer:
            #print(word.getFeat('GOV'))

        sentences_label.append(confo.buffer)

        dependencies_i=dependencies[i]
        print(len(dependencies), i)
        for word in confp.buffer:
            has_new_dep=False
            for dep in dependencies_i:
                if dep[2].getFeat('INDEX')==word.getFeat('INDEX'):
                    has_new_dep=True
                    #print('True')
                    word.setFeat('GOV', dep[0].getFeat('INDEX'))
                    word.setFeat('LABEL', dep[1])
                else:
                    pass
            if not has_new_dep:
                #word.setFeat('GOV', 'unfound')
                word.setFeat('LABEL', 'unfound')
        sentences_predicted.append(confp.buffer)
        i += 1


write_conllu(sentences_label, 'UD_French-GSD/fr_gsd-ud-train.conllu', 'test_conll/fr_gsd-ud-train_label.conllu')
write_conllu(sentences_predicted, 'UD_French-GSD/fr_gsd-ud-train.conllu', 'test_conll/fr_gsd-ud-train_predicted.conllu')





