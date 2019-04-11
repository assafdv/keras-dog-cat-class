import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from create_models import convnet_yann_Lecun
from utils.file_sys_utils import get_nb_files
from train_test_pipeline.constants import IM_HEIGHT, IM_WIDTH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_dog_cat(train_data_dir, validation_data_dir):
    """
    train dog-cat classifier
    :param train_data_dir:
    :param validation_data_dir:
    :return:
    """
    # init
    nb_train_samples = get_nb_files(train_data_dir, suffix='jpg')
    nb_validation_samples = get_nb_files(validation_data_dir, suffix='jpg')

    # configure
    epochs = 50
    batch_size = 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, IM_WIDTH, IM_HEIGHT)
    else:
        input_shape = (IM_WIDTH, IM_HEIGHT, 3)

    # create model
    model_name = 'convnet_yann_Lecun1'
    model = convnet_yann_Lecun(input_shape)

    # compile models
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # show model summary
    output_model_file = os.path.join('models', '{}.h5'.format(model_name))
    model.summary()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

    # fit
    best_model = ModelCheckpoint(output_model_file,
                                 monitor="val_acc",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode="max")
    callbacks = [best_model,
                 TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)]

    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks)

    # model.save(output_model_file)
    plot_training(history_ft)


def plot_training(history):
    """ plot learning curve     """
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(acc))

    plt.plot(epochs, acc, "r.")
    plt.plot(epochs, val_acc, "r")
    plt.title("Training and validation accuracy")
    plt.legend(['train', 'val'])

    plt.figure()
    plt.plot(epochs, loss, "r.")
    plt.plot(epochs, val_loss, "r-")
    plt.title("Training and validation loss")
    plt.legend(['train', 'val'])
    plt.show()


def preview_augmentation(img_file_path, output_dir, nb_iters):
    """
    preview augmentation - save augmented pictures to disk.
    :param img_file_path: string, path to single image file
    :param output_dir: string, path to output dir
    :return:
    """
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(img_file_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_dir, save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > nb_iters:
            break
