from dataParser import readCSV
from resnetJPFer import ResNetPreAct
from myResnet import makeModel

(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
	readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))

print("dataShape: ", trainData.shape)
#print(trainData)

epochs = 100 #default 100
batchSize = 32 #28709/32=897 batches per epoch (fer)
nClasses = len(trainLabels[0]) #7 (fer)

model = makeModel(trainData.shape[1:], #input shape (check channels_first/last)
				  nClasses, #number of classes
				  (128, 3, 2), #Layer1 (Conv2D) args (FilterN, FilterDim, Stride) #TODO: check against paper
				  (32, 3), #bottleneck layer Conv2D args (FilterN, FilterDim)
				  'glorot_normal', #kernel initialiser
				  0.0, #kernel regulariser: l2(reg)
				  25) #nBottlenecks

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

#TODO: loop fit to change learning rate between epochs (DONE IN CIFAR)
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

with open('ferResnetResults.txt', 'w') as f:
	f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
	f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))
