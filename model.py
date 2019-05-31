import os
import time
import cv2
import numpy as np
import load_data as ld

# from keras import layers
# from keras import applications
# from keras.callbacks import ModelCheckpoint
# from keras import models

VGG_WIDTH = VGG_HEIGHT = 224

(x_train,y_train) = ld.load_training_data(save_data=True)
# (x_val,y_val) = ld.load_validation_data(save_data=True)
# x_train = np.load("./x_train.npy")
# y_train = np.load("./y_train.npy")
# x_val = np.load("./x_val.npy")
# y_val = np.load("./y_val.npy")

np.savetxt('save.txt',x_val[0][0])


# vgg_model = applications.VGG16(weights='imagenet',
#                                include_top=False,
#                                input_shape=(VGG_WIDTH, VGG_HEIGHT, 3))

# # Make sure that the pre-trained bottom layers are not trainable
# for layer in vgg_model.layers[:-4]:
#     layer.trainable = False

# # Check the trainable status of the individual layers
# # for layer in vgg_model.layers:
# #     print(layer, layer.trainable)

# # Create the model
# model = models.Sequential()
# model.add(vgg_model)
# # Add new layers
# model.add(layers.Flatten())
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()

# model.compile(loss='binary_crossentropy',
#                      optimizer='adam',
#                      metrics=['accuracy'])

# filepath="./model-logs/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# model.fit(x_train,y_train, epochs = 25,  batch_size=128, validation_data = (x_val,y_val), callbacks=callbacks_list, verbose = 2)

