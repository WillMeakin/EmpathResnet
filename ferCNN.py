from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Conv1D
from keras.models import Model
from dataParser import readCSV

(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
	readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))

print("dataShape: ", trainData.shape)
#print(trainData)

epochs = 1
batchSize = 32 #28709/32=897 batches per epoch (fer)
numClasses = len(trainLabels[0]) #7

inputs = Input(shape=trainData.shape[1:])
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(numClasses)(x)
predictions = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainData, trainLabels,
		  batch_size=batchSize,
		  epochs=epochs,
		  validation_data=(validationData, validationLabels))
evalResult = model.evaluate(testData, testLabels)

print('\n\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)

model.save('ferCNN.h5')
print('model saved.')

with open('ferCNNResults.txt', 'w') as f:
	f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
	f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))

