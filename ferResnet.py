from dataParser import readCSV
from resnetJPFer import ResNetPreAct

(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
	readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))

print("dataShape: ", trainData.shape)
#print(trainData)

epochs = 100 #default 100
batchSize = 32 #28709/32=897 batches per epoch (fer)
numClasses = len(trainLabels[0]) #7 (fer)

model = ResNetPreAct() #TODO: adapt to fer

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

model.fit(trainData, trainLabels,
		  batch_size=batchSize,
		  epochs=epochs,
		  validation_data=(validationData, validationLabels),
		  shuffle=True)

evalResult = model.evaluate(testData, testLabels)

print('\n\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)

model.save('ferRes.h5')
print('model saved.')
