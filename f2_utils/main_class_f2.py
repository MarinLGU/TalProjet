import pickle
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

with open('y_train2Enc.txt', 'rb') as f:
    y_trainEnc=pickle.load(f)
    f.close()



with open('X_train2Enc.txt', 'rb') as f:
    X_train2Enc=pickle.load(f)
    f.close()


with open('y_test2Enc.txt', 'rb') as f:
    y_testEnc=pickle.load(f)
    f.close()

with open('X_test2Enc.txt', 'rb') as f:
    X_test2Enc=pickle.load(f)
    f.close()




model = Sequential()
#model.add(Flatten(input_shape=(178,)))
model.add(BatchNormalization(input_shape=(607,)))
model.add(Dense(500,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200,  activation='relu'))
model.add(BatchNormalization())
model.add(Dense(89, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_train2Enc, y_trainEnc, epochs=10, batch_size=1024)
scoretrain=model.evaluate(X_train2Enc, y_trainEnc, batch_size=1024)
scoretest=model.evaluate(X_test2Enc, y_testEnc, batch_size=1024)
print('\n loss train :', scoretrain[0], 'accuracy train :', scoretrain[1])
print('\n loss test :', scoretest[0], 'accuracy test :', scoretest[1])

# if save:
#     with open('classifier.txt', 'wb') as f:
#         pickle.dump(model, f)
#         f.close()


