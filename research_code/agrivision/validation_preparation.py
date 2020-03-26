import os
import cv2
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import Sequence
from classification_models import Classifiers

from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers import Activation
from keras.models import Model
from keras.backend.tensorflow_backend import set_session


def setup_env():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def parse_args():
    parser = argparse.ArgumentParser(prog='create_validation_set.py')
    parser.add_argument("--data_folder", "-d", type=str, required=True,
                        help="Path to a dataset folder")
    parser.add_argument("--input_csv", "-c", type=str, required=True,
                        help="Path to a csv file with dataset data")
    parser.add_argument("--model", "-m", type=str, required=False,
                        help="Name of the model", default='resnet50')
    parser.add_argument("--epochs", "-e", type=int, required=False,
                        help="Number of epochs", default=5)
    parser.add_argument("--batch", "-b", type=int, required=False,
                        help="Batch size", default=1)
    args = parser.parse_args()
    return args


def make_model(model_name, input_shape=(512, 512, 3), classes=2):
    model_class, preprocess_input = Classifiers.get(model_name)
    pretrained_model = model_class(input_shape=input_shape, include_top=False, weights='imagenet', classes=classes)
    x = GlobalAveragePooling2D(name='pool1')(pretrained_model.output)
    x = Dense(classes, name='fc1')(x)
    x = Activation('softmax', name='softmax')(x)
    model = Model(pretrained_model.input, x)
    return model


class DataSequence(Sequence):
    def __init__(self, img_paths, classes, batch_size):
        self.img_paths = img_paths
        self.classes = classes
        self.batch_size = batch_size

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_imgs_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_classes = np.array([np.array([1, 0]) if class_value == 0 else np.array([0, 1]) for class_value in
                                  self.classes[idx * self.batch_size:(idx + 1) * self.batch_size]])
        batch_imgs = np.array([cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) / 255
                               for path in batch_imgs_paths])
        return batch_imgs, batch_classes


if __name__ == "__main__":
    args = parse_args()
    setup_env()
    model = make_model(args.model)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    train_df = pd.read_csv(args.input_csv)
    train_generator = DataSequence((args.data_folder + train_df["path"]).values, train_df["class"].values,
                                   args.batch)
    os.makedirs("weights", exist_ok=True)
    checkpoint = ModelCheckpoint(f"weights/{args.model}" + "_{epoch:02d}-{acc:.2f}.h5",
                                 monitor='acc',
                                 save_best_only=False,
                                 verbose=1,
                                 period=1)
    model.fit_generator(train_generator, epochs=int(args.epochs), callbacks=[checkpoint])
