from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils import to_categorical
import resnetJPCifar
from myResnet import makeModel

nEpochs = 100
batch_size = 32
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

model = makeModel(trainData.shape[1:], #input shape (check channels_first/last)
				  nClasses, #number of classes
				  (128, 3, 2), #Layer1 (Conv2D) args (FilterN, FilterDim, Stride) #TODO: check against paper
				  (32, 3), #bottleneck layer Conv2D args (FilterN, FilterDim)
				  'glorot_normal', #kernel initialiser
				  0.0, #kernel regulariser: l2(reg)
				  25) #nBottlenecks

opt = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#TODO: loop fit to change learning rate between epochs (TEST)

for epoch in range(nEpochs):
	if epoch == 2: #default 50
		opt.lr = 0.0001
	elif epoch == 3: #default 75
		opt.lr = 0.00001

	print('LR: ', opt.lr, ' at epoch: ', epoch)

	model.fit(trainData, trainLabels,
			  batch_size=batch_size,
			  epochs=1,
			  validation_split=0.15,
			  shuffle=True)

evalResult = model.evaluate(testData, testLabels)

print('\n\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)

model.save('cifarRes.h5')
print('model saved.')

with open('cifarResnetResults.txt', 'w') as f:
	f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
	f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))
