import tensorflow.keras
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import generator

epochs = 100
batch_size = 8
#training_samples = 4000
#validation_samples = 600
training_samples = 400
validation_samples = 200
img_width = 224
img_height = 224
channels = 3
input_shape = (img_width, img_height, 3)

dim = (img_width,img_height)

train_data_dir = 'Small_set_cats_vs_dogs/train'
validation_data_dir = 'Small_set_cats_vs_dogs/val'
test_data_dir = 'Small_set_cats_vs_dogs/test'
#train_datagen = ImageDataGenerator(
#    rescale=1./255
#)

#val_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory(
#    train_data_dir,
#    target_size=(img_width, img_height),
#    batch_size=batch_size,
#    class_mode='binary'
#)

#validation_generator = val_datagen.flow_from_directory(
#    validation_data_dir,
#    target_size=(img_width,img_height),
#    batch_size=batch_size,
#    class_mode='binary'
#)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.4,
    zoom_range=0.4,
    rotation_range=20,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

train_nonaugmented = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

train_non_generator = train_nonaugmented.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=input_shape, padding='same'))
model.add(Conv2D(32,(3,3),activation='relu', input_shape=input_shape, padding='same'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(.25))
model.add(Conv2D(64,(3,3),activation='relu', padding='same'))
model.add(Conv2D(64,(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(.25))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(256, (3,3), activation='relu',padding='same'))
#model.add(Conv2D(256, (3,3), activation='relu',padding='same'))
#model.add(Conv2D(256, (3,3), activation='relu',padding='same'))
#model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(512, (3,3), activation='relu',padding='same'))
#model.add(Conv2D(512, (3,3), activation='relu',padding='same'))
#model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(.25))
model.add(Dense(2, activation='softmax'))
model.summary()

model.load_weights('model_n.h5')
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=0.01)#,decay=1e-6) #,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0,mode='auto',min_delta=0.0001,cooldown=0,min_lr=0)

snapshot_name = 'dummy_model_'
#checkpoint = ModelCheckpoint(filepath=snapshot_name+".{epoch:02d}-{val_acc:.2f}.hdf5",monitor='val_acc',verbose=0, save_best_only=True, save_weights_only=False,mode='auto')

history = model.fit_generator(train_non_generator,
                                  steps_per_epoch = training_samples // batch_size,
                                  epochs = epochs, validation_data = validation_generator,
                                  validation_steps = validation_samples // batch_size
                                  #callbacks=[reduce_lr]
                                  )
model.evaluate(test_generator)
print(history)
model.save_weights('model_new.h5')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#history = model.fit_generator(train_generator,
#                                steps_per_epoch = training_samples // batch_size,
#                                epochs = epochs, validation_data = validation_generator,
#                                validation_steps = validation_samples // batch_size,
#                                callbacks=[reduce_lr])
