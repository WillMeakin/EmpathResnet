from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import cifar10
from dataParser import readCSV


def ferEg():

	(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
		readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))
	model = load_model('ferCNN.h5')

	evalResult = model.evaluate(testData, testLabels)

	print('\nmets: ', model.metrics_names)
	print('evalResult: ', evalResult)

	img = testData[0]

	print(model.predict(img, 1, 1))

def cifarEg():

	nClasses = 10

	# The data, shuffled and split between train and test sets:
	(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

	print('trainData shape:', trainData.shape)
	print(trainData.shape[0], 'train samples')
	print(testData.shape[0], 'test samples')

	# Convert class vectors to binary class matrices.
	trainLabels = to_categorical(trainLabels, nClasses)
	testLabels = to_categorical(testLabels, nClasses)

	trainData = trainData.astype('float32')
	testData = testData.astype('float32')
	trainData /= 255
	testData /= 255

	model = load_model('cifarCNN.h5')

	evalResult = model.evaluate(testData, testLabels)

	print('\nmets: ', model.metrics_names)
	print('evalResult: ', evalResult)

	img = testData[0]

	print(model.predict(img, 1, 1))


#TODO: run against a single image (cifar/fer)