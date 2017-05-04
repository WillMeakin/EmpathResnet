from keras.datasets import cifar10
from keras.utils import to_categorical
import resnetJPCifar

epochs = 100
batch_size = 32
num_classes = 10

# The data, shuffled and split between train and test sets:
(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

print('trainData shape:', trainData.shape)
print(trainData.shape[0], 'train samples')
print(testData.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
trainLabels = to_categorical(trainLabels, num_classes)
testLabels = to_categorical(testLabels, num_classes)

trainData = trainData.astype('float32')
testData = testData.astype('float32')
trainData /= 255
testData /= 255

model = resnetJPCifar.ResNetPreAct()
#model = myResnet.makeModel(trainData.shape[1:])

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

model.save('cifarRes.h5')
print('model saved.')
