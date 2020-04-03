import os
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_applications import imagenet_utils
from research_code.params import args
from sklearn.utils import shuffle as skl_shuffle
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import albumentations as albu


def reshape(reshape_size=(512,512)):
    return albu.Compose([albu.Resize(reshape_size[0], reshape_size[1], p=1)], p=1)


def strong_aug(crop_size=(512,512)):
    return albu.Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Transpose(),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
        ], p=0.2),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        albu.OneOf([
            albu.OpticalDistortion(p=0.3),
            albu.GridDistortion(p=0.1),
            albu.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomContrast(),
            albu.RandomBrightness(),
            albu.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ], p=0.3),
        albu.HueSaturationValue(p=0.3),
        albu.RandomCrop(crop_size[0], crop_size[1], p=1),
    ], p=1)


class DataGenerator_agrivision(tf.keras.utils.Sequence):
    """Generates data for Keras
    """

    def __init__(
        self,
        dataset_df,
        classes,
        img_dir=None,
        batch_size=None,
        shuffle=False,
        reshape_size=None,
        crop_size=None,
        do_aug=False,
    ):
        "Initialization"
        self.dataset_df = dataset_df
        self.classes = classes
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reshape_size = reshape_size
        self.crop_size = crop_size
        self.do_aug = do_aug
        self.aug = strong_aug(crop_size)
        self.reshape_func = reshape(reshape_size)
        
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.dataset_df) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        """
        # Generate indexes of the batch
        batch_data = self.dataset_df[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Generate data
        X, y = self.__data_generation(batch_data)

        return X, y

    def on_epoch_end(self):
        """Shuffle the dataset on epoch end
        """
        if self.shuffle == True:
            skl_shuffle(self.dataset_df)

    def read_masks_borders(self, name):
        border_path = os.path.join(
            self.img_dir, "boundaries", name.replace(".jpg", ".png")
        )
        border_img = cv2.imread(border_path)
        mask_path = os.path.join(self.img_dir, "masks", name.replace(".jpg", ".png"))
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path)
        else:
            mask_img = np.ones(border_img.shape)

        mask_img = mask_img * border_img
        mask_img = mask_img > 0
        mask_img = np.invert(mask_img)
        return mask_img

    def __data_generation(self, batch_data):
        """Generates data containing batch_size samples 
           X : (n_samples, *dim, n_channels)
        """
        # Initialization
        batch_x = []
        batch_y = defaultdict(list)

        for ind, filename in batch_data.iterrows():
            img_path = os.path.join(self.img_dir, "images", "rgb", filename["name"])
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as error:
                print(img_path)
                print(error)
            not_valid_mask = self.read_masks_borders(filename["name"])
            img[not_valid_mask] = 0

            # getmasks
            targets = np.zeros((img.shape[0], img.shape[1], len(self.classes)))
            for i, c in enumerate(self.classes):
                mask_path = os.path.join(self.img_dir, "labels", c, filename["name"])
                mask = cv2.imread(
                    mask_path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE
                )
                mask[not_valid_mask[:, :, 0]] = 0
                mask = mask > 0
                targets[:,:,i] = mask 

            res = self.reshape_func(image=img, mask=targets)
            img, targets = res['image'], res['mask']
            if self.do_aug:
                res = self.aug(image=img, mask=targets)
                img, targets = res['image'], res['mask']

            # FIXME: Remove as soon as moved to new models
            for i, c in enumerate(self.classes):
                batch_y[c].append(targets[:,:,i])
            
            batch_x.append(img)
            # batch_y.append(targets)
        
            
        batch_x = np.array(batch_x, np.float32)
        batch_y = {k: np.array(v, np.float32) for k, v in batch_y.items()}
        batch_y = {k: np.expand_dims(v, axis=-1) for k, v in batch_y.items()}
        # FIXME: Uncomment when moved to new models
        # batch_y = np.array(batch_y, np.float32)

        return (
            imagenet_utils.preprocess_input(batch_x, "channels_last", mode="tf"),
            batch_y
        )
