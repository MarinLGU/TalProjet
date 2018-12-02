from keras.models import Sequential
from keras.layers import Dense, Flatten


model = Sequential()
#model.add(Flatten(input_shape=(178,)))
model.add(Dense(178,  activation='relu', input_shape=(178,)))
model.add(Dense(89, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit()