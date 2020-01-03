from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.layers as l
from tensorflow.keras import optimizers


class _CNN1_Model(Sequential):
	default_optimizer = optimizers.Adam(learning_rate=0.001)

	def compile(self, optimizer=default_optimizer, loss='categorical_crossentropy', **kwargs):
		return super().compile(optimizer, loss=loss, **kwargs)


def CNN1(num_classes, input_shape=(128, 128, 3)):
	model = _CNN1_Model()

	# Image input
	model.add(l.Input(shape=input_shape))
	
	# Convolutional block
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))

	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))

	model.add(l.Conv2D(64, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))

	# Feature extraction and evaluation
	model.add(l.Flatten())
	model.add(l.Dense(64))
	model.add(l.Activation('relu'))
	model.add(l.Dropout(0.5))
	model.add(l.Dense(num_classes))
	model.add(l.Activation('softmax'))

	return model


class _AMDNet_Model(Sequential):
	default_optimizer = optimizers.Adam(learning_rate=0.0001)

	def compile(self, optimizer=default_optimizer, loss='categorical_crossentropy', **kwargs):
		return super().compile(optimizer, loss=loss, **kwargs)


def AMDNet(num_classes, input_shape=(512, 512, 3)):
	model = _AMDNet_Model()

	# Image input
	model.add(l.Input(shape=input_shape))
	
	# Convolutional blocks
	model.add(l.Conv2D(64, (5, 5)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	model.add(l.Conv2D(64, (3, 3)))
	model.add(l.Activation('relu'))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	
	model.add(l.Conv2D(32, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	
	model.add(l.Conv2D(64, (3, 3)))
	model.add(l.Activation('relu'))
	
	model.add(l.Conv2D(64, (3, 3)))
	model.add(l.Activation('relu'))
	model.add(l.MaxPooling2D(pool_size=(2, 2)))
	
	# Feature extraction and evaluation
	model.add(l.Flatten())
	model.add(l.Dense(64))
	model.add(l.Activation('relu'))
	model.add(l.Dropout(0.5))
	model.add(l.Dense(64))
	model.add(l.Activation('relu'))
	model.add(l.Dropout(0.2))
	model.add(l.Dense(num_classes))
	model.add(l.Activation('softmax'))

	return model


class _GONNet_Model(Model):
	default_optimizer = optimizers.Adam(learning_rate=0.0001)

	def compile(self, optimizer=default_optimizer, loss='categorical_crossentropy', **kwargs):
		return super().compile(optimizer, loss=loss, **kwargs)

def GONNet(num_classes, input_shape=(224, 224, 3)):
	# ResNet50 architecture
	renet50_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
	x = renet50_model.output
	
	# Feature extraction and evaluation
	x = l.Flatten()(x)
	x = l.Dropout(0.5)(x)
	x = l.Dense(64, activation='relu')(x)
	x = l.Dropout(0.2)(x)
	x = l.Dense(num_classes, activation='softmax')(x)

	model = _GONNet_Model(inputs=renet50_model.input, outputs=x)
	
	return model


if __name__ == '__main__':
	print("CNN1 model architecture summary")
	cnn1_model = CNN1(4)
	cnn1_model.compile()
	cnn1_model.summary()

	print("")
	print("AMDNet model architecture summary")
	amdnet_model = AMDNet(2)
	amdnet_model.compile()
	amdnet_model.summary()

	print("")
	print("GONNet model architecture summary")
	gonnet_model = GONNet(2)
	gonnet_model.compile()
	gonnet_model.summary()
