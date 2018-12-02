import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import pickle


save=False
langue='English'
with open('y_train1Enc%s.txt'%langue, 'rb') as f:
    y_trainEnc=pickle.load(f)
    f.close()

with open('X_train1Enc%s.txt'%langue, 'rb') as f:
    X_train1Enc=pickle.load(f)
    f.close()

with open('y_test1Enc%s.txt'%langue, 'rb') as f:
    y_testEnc=pickle.load(f)
    f.close()

with open('X_test1Enc%s.txt'%langue, 'rb') as f:
    X_test1Enc=pickle.load(f)
    f.close()




model = Sequential()
#model.add(Flatten(input_shape=(178,)))
model.add(BatchNormalization(input_shape=(X_train1Enc.shape[1],)))
model.add(Dense(500,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(89, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_train1Enc, y_trainEnc, epochs=10, batch_size=1024)
scoretrain=model.evaluate(X_train1Enc, y_trainEnc, batch_size=1024)
scoretest=model.evaluate(X_test1Enc, y_testEnc, batch_size=1024)
print('\n loss train :', scoretrain[0], 'accuracy train :', scoretrain[1])
print('\n loss test :', scoretest[0], 'accuracy test :', scoretest[1])


if save:
    with open('classifier.txt', 'wb') as f:
        pickle.dump(model, f)
        f.close()





