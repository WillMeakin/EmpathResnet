from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input

batch_size = 32
num_classes = 10
epochs = 1

# The data, shuffled and split between train and test sets:
(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

print(trainData.shape[0], 'train samples')
print(testData.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
trainLabels = keras.utils.to_categorical(trainLabels, num_classes)
testLabels = keras.utils.to_categorical(testLabels, num_classes)

print('trainData shape:', trainData.shape)
print('cifarPrints: ', type(trainData), type(trainData[0]), type(trainData[0][0]), type(trainData[0][0][0]), type(trainData[0][0][0][0]))
print(len(trainData))
print(len(trainData[0]))
print(len(trainData[0][0]))
print(len(trainData[0][0][0]))
#print(trainData)

trainData = trainData.astype('float32')
testData = testData.astype('float32')
trainData /= 255
testData /= 255

inputs = Input(shape=trainData.shape[1:])
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(num_classes)(x)
predictions = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(trainData, trainLabels,
		  batch_size=batch_size,
		  epochs=epochs,
		  validation_split=0.15,
		  shuffle=True)
evalResult = model.evaluate(testData, testLabels)

print('\n\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)

model.save('cifarCNN.h5')
print('model saved.')

with open('cifarCNNResults.txt', 'w') as f:
	f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
	f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))
