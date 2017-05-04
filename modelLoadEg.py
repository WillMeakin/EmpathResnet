from keras.models import Model, load_model
from dataParser import readCSV

(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
	readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))


model = load_model('ferCNN.h5')

evalResult = model.evaluate(testData, testLabels)

print('\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)