from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Activation, BatchNormalization, add, AveragePooling2D, Flatten, Dense
from keras.regularizers import l2
from keras.backend import image_data_format

#CBACBACM
def bottleneck(inTensor, L1Args, LBArgs, kernelInit, kernelReg, axis):

	x = Conv2D(LBArgs[0], #Number of filters
				(1, 1), #Filter dim
				#strides=(,), #default (1,1)
				#padding='', #default  'valid'
				use_bias=False,
				kernel_initializer=kernelInit,
				kernel_regularizer=l2(kernelReg))(x)
	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(LBArgs[0], #Number of filters
				(LBArgs[1], LBArgs[1]), #Filter dim
				#strides=(,), #default (1,1)
				padding='same',
				use_bias=False,
				kernel_initializer=kernelInit,
				kernel_regularizer=l2(kernelReg))(x)
	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(L1Args[0], #Number of filters
				(1, 1), #Filter dim
				#strides=(,), #default (1,1)
				#padding='', #default  'valid'
				use_bias=False,
				kernel_initializer=kernelInit,
				kernel_regularizer=l2(kernelReg))(x)

	x = add([x, inTensor]) #Merge identity and bottleneck tensors
	return x

#L1Args and LBArgs = (FilterN, FilterDim, Stride)
def makeModel(inShape, nClasses, L1Args, LBArgs, kernelInit, kernelReg, nBottlenecks):

	axis = -1 #Used for batch normalisation
	if image_data_format() == 'channels_first':
		axis = 1
	elif image_data_format() == 'channels_last':
		axis = 3	#TODO: IS THIS CORRECT? print conv2d output

	inputs = Input(shape=inShape)

	x = Conv2D(L1Args[0], #Number of filters
				(L1Args[1], L1Args[1]), #Filter dim
				strides=(L1Args[2], L1Args[2]),
				padding='same',
				use_bias=False,
				kernel_initializer=kernelInit, #TODO: why use this?
				kernel_regularizer=l2(kernelReg))(inputs) #TODO: why use this?
	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)

	#First bottleneck has no BatchNorm or Activation
	x = bottleneck(x, L1Args, LBArgs, kernelInit, kernelReg, axis)
	for i in range(nBottlenecks):
		x = BatchNormalization(axis=axis)(x)
		x = Activation('relu')(x)
		x = bottleneck(x, L1Args, LBArgs, kernelInit, kernelReg, axis)

	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)

	#TODO: use final convolution?

	poolSize = inShape[1]/L1Args[2]
	x = AveragePooling2D((poolSize, poolSize))(x)

	x = Flatten()(x)
	predictions = Dense(nClasses, activation='softmax')

	return Model(input=inputs, outputs=predictions)