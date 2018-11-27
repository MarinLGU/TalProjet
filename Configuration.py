from Transitions import *
class Configuration():
    def __init__(self, WordBuffer): #s est la phrase
        self.buffer=[word for word in WordBuffer.nextSentence()]
        self.stack=[]
        self.dependencies=[]









