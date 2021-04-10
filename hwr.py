from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator as IDG
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0

unique, counts = np.unique(yTrain, return_counts=True)
data = {'Digits': [i for i in unique], 'Number of images': [i for i in counts]}
df = pd.DataFrame(data)
sns.barplot(x='Digits', y='Number of images', data=df)


def showIm(n):
    plt.figure()
    plt.imshow(np.array(xTrain[n]).reshape((28, 28)), cmap='binary_r')
    plt.xlabel(f"Value: {yTrain[n]}", fontsize=18)
    plt.show()


showIm(randint(0, 10000))

xTrain = np.reshape(xTrain, (-1, 28, 28, 1))
print(xTrain.shape)

dataGeneration = IDG(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

trainDataGen = dataGeneration.flow(xTrain, yTrain, batch_size=64, subset='training')
valDataGen = dataGeneration.flow(xTrain, yTrain, batch_size=64, subset='validation')

model = Sequential()

model.add(Conv2D(16, 8, input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(32, 6, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(trainDataGen, validation_data=valDataGen, epochs=30)

pred = model.predict(np.reshape(xTest, (-1, 28, 28, 1)))
yPred = [np.argmax(x) for x in pred]

c = len(yPred)
for i in range(len(yPred)):
    if yPred[i] != yTest[i]:
        c -= 1

print(f"Predicted Accuracy on Test set: {c * 100 / len(yPred)}%")

path = "hwr.h5"
model.save(path)
