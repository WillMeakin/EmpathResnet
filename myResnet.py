from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Activation


def makeModel(shape, ):

	inputs = Input(shape=shape)

	x = Conv2D(32, (3, 3), padding='same')(inputs)

	predictions = Activation('softmax')(x)

	model = Model(input=inputs, outputs=predictions)