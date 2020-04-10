import os
import keras
import cv2

import numpy as np
import tensorflow as tf
import albumentations as albu

from typing import List
from collections import defaultdict
from keras_applications import imagenet_utils
from sklearn.utils import shuffle as skl_shuffle
from imblearn.keras import BalancedBatchGenerator
from imblearn.over_sampling import RandomOverSampler


def reshape(reshape_size=(512, 512)):
    return albu.Compose([albu.Resize(reshape_size[0], reshape_size[1], interpolation=cv2.INTER_NEAREST, p=1)], p=1)


def strong_aug(crop_size=(512, 512)):
    return albu.Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Transpose(),
        # albu.OneOf([
        #    albu.IAAAdditiveGaussianNoise(),
        #   albu.GaussNoise(),
        # ], p=0.2),
        # albu.OneOf([
        #    albu.MotionBlur(p=0.2),
        #    albu.MedianBlur(blur_limit=3, p=0.1),
        #    albu.Blur(blur_limit=3, p=0.1),
        # ], p=0.2),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # albu.OneOf([
        #     albu.OpticalDistortion(p=0.3),
        #     albu.GridDistortion(p=0.1),
        #     albu.IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        albu.OneOf([
            # albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomContrast(),
            albu.RandomBrightness(),
            # albu.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ], p=0.3),
        # albu.HueSaturationValue(p=0.3),
        albu.RandomCrop(crop_size[0], crop_size[1], p=1),
    ], p=1)


def read_gray_scale_img(full_img_path: str) -> np.ndarray:
    return cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)


def read_channels(channels: List[str], img_name: str, img_dir: str) -> np.ndarray:
    channel_stack = []
    use_rgb = False
    possible_channels = ['r', 'g', 'b', 'nir', 'ndvi', 'ndwi', 'l']
    rgb_channels = ['r', 'g', 'b']
    for rgb_channel in rgb_channels:
        if rgb_channel in channels:
            use_rgb = True
    rgb = None
    if use_rgb:
        rgb_path = os.path.join(img_dir, "images", "rgb", img_name)
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    for channel in channels:
        if channel in possible_channels:
            if channel in rgb_channels:
                channel_idx = rgb_channels.index(channel)
                channel_img = rgb[:, :, channel_idx]
                channel_stack.append(channel_img)
            else:
                channel_img = read_gray_scale_img(os.path.join(img_dir, "images", channel, img_name))
                channel_stack.append(channel_img)
        else:
            raise ValueError(f"Unknown channel {channel}")
    channel_stack = np.dstack(channel_stack)
    return channel_stack


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
            use_oversampling=False
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
        self.use_oversampling = use_oversampling
        self.index_generator = None
        self.single_class_samples = None
        if self.use_oversampling:
            # added in case only part of dataframe is used and indexes should be reset for proper indexing
            self.dataset_df = self.dataset_df.reset_index()
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        """
        if self.use_oversampling:
            return len(self.index_generator)
        else:
            return int(np.floor(len(self.dataset_df) / self.batch_size))

    def convert_multilabel_df_to_single_class_samples(self, df):
        pick_class_df = df.copy()
        pick_class_df[self.classes] = (pick_class_df[self.classes] != 0).astype(int)
        total_classes_df = pick_class_df[self.classes].sum().sort_values(ascending=True)
        total_classes_df = total_classes_df.reset_index().set_index("index")
        for class_name in self.classes:
            class_prior = len(self.classes) - np.where(total_classes_df.index.values == class_name)[0][0]
            pick_class_df[class_name] *= class_prior
        samples = pick_class_df[self.classes].max(axis=1)
        return samples

    def get_classes_indices(self, samples):
        img_classes = keras.utils.to_categorical(samples.values - 1, len(self.classes))
        indices = samples.index.values.reshape(-1, 1)
        return img_classes, indices

    def sample_data(self, index):
        if self.use_oversampling:
            indices, classes = self.index_generator[index]
            indices = np.squeeze(indices)
            return self.dataset_df.iloc[indices]
        else:
            return self.dataset_df[index * self.batch_size: (index + 1) * self.batch_size]

    def __getitem__(self, index):
        """Generate one batch of data
        """
        batch_data = self.sample_data(index)

        # Generate data
        X, y = self._data_generation(batch_data)

        return X, y

    def on_epoch_end(self):
        """Shuffle the dataset on epoch end
        """
        if self.use_oversampling:
            if self.single_class_samples is None:
                self.single_class_samples = self.convert_multilabel_df_to_single_class_samples(self.dataset_df)
            self.single_class_samples = skl_shuffle(self.single_class_samples)
            img_classes, indices = self.get_classes_indices(self.single_class_samples)
            sampler = RandomOverSampler(sampling_strategy='not majority')
            self.index_generator = BalancedBatchGenerator(indices, img_classes, sampler=sampler,
                                                          batch_size=self.batch_size,
                                                          random_state=42)
        elif self.shuffle:
            self.dataset_df = skl_shuffle(self.dataset_df)

    def read_masks_borders(self, name):
        border_path = os.path.join(
            self.img_dir, "boundaries", name.replace(".jpg", ".png")
        )

        border_img = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
        try:
            border_img = border_img > 0
        except Exception as error:
            print(border_path)
        border_img = np.invert(border_img)
        return border_img

    def _data_generation(self, batch_data):
        """Generates data containing batch_size samples 
           X : (n_samples, *dim, n_channels)
        """
        # Initialization
        batch_x = []
        batch_y = defaultdict(list)

        for ind, item_data in batch_data.iterrows():
            img_path = os.path.join(self.img_dir, "images", "rgb", item_data["name"])
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as error:
                print(img_path)
                print(error)
            not_valid_mask = self.read_masks_borders(item_data["name"])
            img[not_valid_mask] = 0

            # getmasks
            targets = np.zeros((img.shape[0], img.shape[1], len(self.classes)))
            for i, c in enumerate(self.classes):
                mask_path = os.path.join(self.img_dir, "labels", c, item_data["name"])
                mask = cv2.imread(
                    mask_path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE
                )
                mask[not_valid_mask[:, :, 0]] = 0
                mask = mask > 0
                targets[:, :, i] = mask

            res = self.reshape_func(image=img, mask=targets)
            img, targets = res['image'], res['mask']
            if self.do_aug:
                res = self.aug(image=img, mask=targets)
                img, targets = res['image'], res['mask']

            for i, c in enumerate(self.classes):
                batch_y[c].append(targets[:, :, i])

            batch_x.append(img)

        batch_x = np.array(batch_x, np.float32)
        batch_y = {k: np.array(v, np.float32) for k, v in batch_y.items()}
        batch_y = {k: np.expand_dims(v, axis=-1) for k, v in batch_y.items()}

        return (
            imagenet_utils.preprocess_input(batch_x, "channels_last", mode="tf"),
            batch_y
        )


class DataGeneratorSingleOutput(DataGenerator_agrivision):
    'Generates data for Keras'

    def __init__(self,
                 dataset_df,
                 classes,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 reshape_size=None,
                 crop_size=None,
                 do_aug=False,
                 activation=None,
                 validate_pixels=True,
                 channels=None,
                 use_oversampling=False):
        'Initialization'
        super().__init__(dataset_df, classes, img_dir, batch_size, shuffle, reshape_size, crop_size, do_aug,
                         use_oversampling)

        if activation is None:
            raise ValueError("Please pick activation function!")
        self.activation = activation
        self.validate_pixels = validate_pixels
        self.channels = channels

    def _data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        train_batch = list_IDs_temp
        batch_x = []
        batch_y = []

        for ind, item_data in train_batch.iterrows():
            channels = read_channels(self.channels, item_data["name"], self.img_dir)

            if self.validate_pixels:
                not_valid_mask = self.read_masks_borders(item_data['name'])
            else:
                not_valid_mask = np.zeros((channels.shape[0], channels.shape[1]), dtype=np.bool)

            channels[not_valid_mask] = 0
            targets = np.zeros((channels.shape[0], channels.shape[1], len(self.classes)))
            for idx, cls in enumerate(self.classes):
                mask_path = os.path.join(self.img_dir, 'labels', cls, item_data['name'])
                mask = cv2.imread(mask_path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
                mask[not_valid_mask] = 0
                mask = mask > 0
                targets[:, :, idx] = mask

            res = self.reshape_func(image=channels, mask=targets)
            channels, targets = res['image'], res['mask']
            if self.do_aug:
                res = self.aug(image=channels, mask=targets)
                channels, targets = res['image'], res['mask']
            batch_y.append(targets)
            batch_x.append(channels)

        batch_x = np.array(batch_x, np.float32)
        batch_y = np.array(batch_y, np.float32)

        if self.activation == 'softmax':
            # the class with higher value gets picked if several classes are present
            # the following coefs were handpicked
            class_priorities = {
                "background": 1,
                'weed_cluster': 2,
                'waterway': 3,
                'standing_water': 4,
                'double_plant': 5,
                'planter_skip': 6,
                'cloud_shadow': 7
            }
            for idx, cls in enumerate(self.classes):
                batch_y[:, :, :, idx] *= class_priorities[cls]
            # (height, width, classes) -> (height, width)
            highest_score_label = batch_y.argmax(axis=-1)
            # (height, width) -> (height, width, classes)
            batch_y = tf.one_hot(highest_score_label, len(self.classes), dtype=np.float32).numpy()

        return imagenet_utils.preprocess_input(batch_x, 'channels_last', mode='tf'), batch_y


class DataGeneratorClassificationHead(DataGeneratorSingleOutput):
    'Generates data for Keras'

    def __init__(self,
                 dataset_df,
                 classes,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 reshape_size=None,
                 crop_size=None,
                 do_aug=False,
                 activation=None,
                 validate_pixels=True,
                 channels=None,
                 use_oversampling=False):
        'Initialization'
        super().__init__(dataset_df, classes, img_dir, batch_size, shuffle, reshape_size, crop_size, do_aug, activation,
                         validate_pixels, channels, use_oversampling)

    def _data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        batch_x, batch_y = super()._data_generation(list_IDs_temp)
        classes = np.array(np.count_nonzero(batch_y, axis=(1, 2)) != 0, dtype=np.float32)
        if 'background' in self.classes:
            classes = classes[:, :-1]
        return batch_x, [batch_y, classes]
