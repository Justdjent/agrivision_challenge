import random

import cv2
import numpy as np
import pandas as pd
from numba import jit
import numpy as np
from keras_applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from research_code.params import args
from sklearn.model_selection import train_test_split
import sklearn.utils
from research_code.random_transform_mask import ImageWithMaskFunction, pad_img, tiles_with_overlap, read_img_opencv, rgb2rgg, get_window
from research_code.random_transform_mask import pad as pad_folder
import os

@jit(nopython=True)
def go_fast(dist_map, size, cat_number):
    res_arr = np.zeros((size[0] * size[1], cat_number))
    flatten_map = dist_map.flatten()
    for i in np.arange(flatten_map.shape[0]):
        arr_x = np.arange(0, cat_number)
        mu = flatten_map[i]
        sigma = 1.5
        res = np.exp(-np.square(arr_x - mu)/ ( 2 * np.square(sigma))) / (sigma * np.sqrt(2*np.pi))
        res = res/res.max()
        res_arr[i] = res
    return res_arr


def prepare_dist(mask_path, size, max_disst=61, mask_between_rows=True):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, size, cv2.INTER_NEAREST)
    line_mask = mask > 200
    dist_mask = mask.astype(np.float32) * (max_disst - 1) / 255
    #res_arr = []
    res_arr = go_fast(dist_mask, size, cat_number=max_disst)
    result = np.array(res_arr).reshape((size[0], size[1], max_disst))
    return result, line_mask


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 filenames,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 out_size=None,
                 crop_size=None,
                 aug=False):
        'Initialization'
        self.filenames = filenames
        self.out_size = out_size
        self.batch_size = batch_size
        self.labels = filenames
        self.img_dir = img_dir
        self.aug = aug
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mask_function = ImageWithMaskFunction(out_size=out_size,
                                                   crop_size=crop_size,
                                                   mask_dir=img_dir)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.filenames = np.arange(len(self.filenames))
        if self.shuffle == True:
            sklearn.utils.shuffle(self.filenames)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        train_batch = list_IDs_temp
        batch_x = []
        weights = []
        for ind, filename in train_batch.iterrows():
            img_path = os.path.join(
                self.img_dir, str(filename['folder']), 'images', filename['name'],
            )
            img = img_to_array( load_img(img_path, grayscale=False))
            if img.shape[:2] != self.out_size:
                img, mask_img = pad_img(img, None, self.out_size)

            batch_x.append(img)
            weights.append(filename["weight"])

        batch_x = np.array(batch_x, np.float32)
        batch_x, masks, probs, edges = self.mask_function.mask_pred(
            batch_x,
            train_batch,
            # range(self.batch_size),
            self.img_dir,
            self.aug,
        )
        weights = np.array(weights)
        if self.crop_size is None:
            # TODO: Remove hardcoded padding
            batch_x, masks = pad(batch_x, 1, 0), pad(masks, 1, 0)

        return imagenet_utils.preprocess_input(
            batch_x, 'channels_last', mode='tf'),\
               {"mask": masks}
            # "instance": probs}


class DataGenerator_lines(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 filenames,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 out_size=None,
                 crop_size=None,
                 aug=False):
        'Initialization'
        self.filenames = filenames
        self.out_size = out_size
        self.batch_size = batch_size
        self.labels = filenames
        self.img_dir = img_dir
        self.aug = aug
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mask_function = ImageWithMaskFunction(out_size=out_size,
                                                   crop_size=crop_size,
                                                   mask_dir=img_dir)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.filenames = np.arange(len(self.filenames))
        if self.shuffle == True:
            sklearn.utils.shuffle(self.filenames)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        train_batch = list_IDs_temp
        batch_x = []
        weights = []
        for ind, filename in train_batch.iterrows():
            img_path = os.path.join(
                self.img_dir, str(filename['folder']), 'images', filename['name'],
            )
            img = img_to_array( load_img(img_path, grayscale=False))
            if img.shape[:2] != self.out_size:
                img, mask_img = pad_img(img, None, self.out_size)

            batch_x.append(img)
            weights.append(filename["weight"])

        batch_x = np.array(batch_x, np.float32)
        batch_x, masks, probs, edges = self.mask_function.mask_pred(
            batch_x,
            train_batch,
            # range(self.batch_size),
            self.img_dir,
            self.aug,
        )
        weights = np.array(weights)
        return imagenet_utils.preprocess_input(
            batch_x, 'channels_last', mode='tf'),\
               {"line_tree": masks,
                "line_gap": probs}


class DataGenerator_angles(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 filenames,
                 classes,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 out_size=None,
                 crop_size=None,
                 aug=False):
        'Initialization'
        self.filenames = filenames
        self.out_size = out_size
        self.batch_size = batch_size
        self.labels = filenames
        self.img_dir = img_dir
        self.aug = aug
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.classes = classes
        self.mask_function = ImageWithMaskFunction(out_size=out_size,
                                                   crop_size=crop_size,
                                                   mask_dir=img_dir)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.filenames = np.arange(len(self.filenames))
        if self.shuffle == True:
            sklearn.utils.shuffle(self.filenames)

    def read_masks_borders(self, name):
        border_path = os.path.join(self.img_dir, 'boundaries', name.replace(".jpg", ".png"))
        border_img = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
        mask_path = os.path.join(self.img_dir, 'masks', name.replace(".jpg", ".png"))
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask_img = np.ones(border_img.shape)
        
        mask_img = mask_img * border_img
        mask_img = mask_img > 0
        mask_img = np.invert(mask_img)
        return mask_img


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        train_batch = list_IDs_temp
        batch_x = []
        class_labels = {cls: [] for cls in self.classes}

        for ind, filename in train_batch.iterrows():
            img_path = os.path.join(
                self.img_dir, 'images', "rgb", filename['name'])
            nir_img_path = os.path.join(
                self.img_dir, 'images', "nir", filename['name'])
            img = cv2.imread(img_path)
            nir_img = cv2.imread(nir_img_path, cv2.IMREAD_GRAYSCALE)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab = lab[:, :, 0]
            lab = np.expand_dims(lab, axis=-1)
            nir_img = np.expand_dims(nir_img, axis=-1)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as error:
                print(img_path)
                print(error)
            img = np.concatenate([img, lab, nir_img], axis=-1)
            not_valid_mask = self.read_masks_borders(filename['name'])
            img[not_valid_mask] = 0
            img = cv2.resize(img, self.out_size)

            # getmasks
            for cls in self.classes:
                mask_path = os.path.join(self.img_dir, 'labels', cls, filename['name'])
                mask = cv2.imread(mask_path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
                mask = mask > 0
                # print(mask.shape)
                class_labels[cls].append(mask)

            batch_x.append(img)

        batch_x = np.array(batch_x, np.float32)
        for cls in self.classes:
            class_labels[cls] = np.expand_dims(np.array(class_labels[cls], np.float32), axis=-1)
        #     masks = np.array(masks, np.float32)
        # line_masks = np.array(line_masks, np.float32)
        # line_masks = np.expand_dims(line_masks, axis=-1)
        # mask_pred_angles(self, batch_x, masks, aug=False)
        batch_x, masks = self.mask_function.mask_pred_angles(
            batch_x,
            class_labels,
            self.classes,
            self.aug,
        )
        return imagenet_utils.preprocess_input(batch_x, 'channels_last', mode='tf'), masks


class DataGenerator_lstm(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 filenames,
                 img_dir=None,
                 batch_size=None,
                 shuffle=False,
                 out_size=None,
                 crop_size=None,
                 aug=False,
                 seq_length=15):
        'Initialization'
        self.filenames = filenames
        self.out_size = out_size
        self.batch_size = batch_size
        self.labels = filenames
        self.img_dir = img_dir
        self.aug = aug
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.mask_function = ImageWithMaskFunction(out_size=out_size,
                                                   crop_size=crop_size,
                                                   mask_dir=img_dir)
        self.seq_length = seq_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames['line_num'].unique()) / self.batch_size))

    def __getitem__(self, index):
        #x, y = super()[index]
        'Generate one batch of data'
        # Generate indexes of the batch
        # list(range(index*self.batch_size, (index+1)*self.batch_size))
        #indexes = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        #indexes = list(range(index*self.batch_size, (index+1)*self.batch_size))
        # Find list of IDs
        #list_IDs_temp = [self.filenames[k] for k in indexes]
        indexes = [3]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.filenames = np.arange(len(self.filenames))
        if self.shuffle == True:
            sklearn.utils.shuffle(self.filenames)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #train_batch = list_IDs_temp
        batch_x = []
        weights = []
        sequences_x = []
        sequences_y = []
        for line_num in list_IDs_temp:
            train_batch = self.filenames[self.filenames['line_num'] == line_num]
            img_path = os.path.join(
                self.img_dir, str(train_batch['folder'].values[0]), 'images', train_batch['img_name'].values[0])
            img = cv2.imread(img_path)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as error:
                print(img_path)
                print(error)
            img_side = img.shape[0]
            img = cv2.resize(img, self.out_size)
            full_sequence = convert_points(img_side, train_batch, output_size=32)
            full_sequence = np.expand_dims(full_sequence, axis=-1)
            x_sequence = full_sequence[:-1]
            y_sequence = full_sequence[1:]
            num_pts_in_line = y_sequence.shape[0]

            # pad x
            net_input_x = np.zeros((self.seq_length, 32, 32, 1))
            net_input_x[:num_pts_in_line] = x_sequence

            # pad y
            net_input_y = np.zeros((self.seq_length, 32, 32, 1))
            net_input_y[:num_pts_in_line] = y_sequence
            # batch_x.append(np.repeat(np.expand_dims(img, axis=0), self.seq_length, axis=0))
            batch_x.append(img)
            sequences_x.append(net_input_x)
            sequences_y.append(net_input_y)

        batch_x = np.array(batch_x, np.float32)
        batch_x = imagenet_utils.preprocess_input(batch_x, 'channels_last', mode='tf')
        sequences_x = np.array(sequences_x, np.float32)
        sequences_y = np.array(sequences_y, np.float32)
        # mask_pred_angles(self, batch_x, masks, aug=False)
        # batch_x, masks = self.mask_function.mask_pred_angles(
        #     batch_x,
        #     masks,
        #     self.aug,
        # )
        # if True:
        #     visualize_seq(sequences_x, sequences_y)
        return {"image_input": batch_x,
                "point_input": sequences_x}, \
               {"point_pred": sequences_y}


def visualize_seq(preds, sequences_y, thresh=0.1):
    for img, gt in zip(preds, sequences_y):

        images_stack = img[:, :, :, 0]
        #images_stack = (images_stack > thresh).astype(np.float32)
        gt_stack = gt[:, :, :, 0]
        image_to_plot = np.zeros((256, images_stack.shape[0] * 128))
        #gt_to_plot = np.zeros((images_stack.shape[0] * 128, 128))
        # batch_img, batch_mask, weights = next(generator)
        # batch_img = batch_img.reshape((batch_img.shape[0] * batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]))
        # batch_img = (((batch_img + 1) / 2) * 255).astype('uint8')
        # batch_mask = batch_mask.reshape((batch_mask.shape[0] * batch_mask.shape[1], batch_mask.shape[2], batch_mask.shape[3]))
        # batch_mask = (batch_mask * 255).astype('uint8')
        #cols = []
        for i in range(images_stack.shape[0]):
            image_to_plot[:128, i * 128: (i+1) * 128] = cv2.resize(images_stack[i], (128, 128))
            image_to_plot[128:, i * 128: (i + 1) * 128] = cv2.resize(gt_stack[i], (128, 128))

        cv2.imshow('msk', image_to_plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def convert_points(img_side, image_dataframe, output_size=32):
    conv_pts = []
    # image_dataframe[image_dataframe['line_num'] == 1]
    for num, row in image_dataframe.iterrows():
        out_side = 32
        scale_factor = img_side / out_side
        center = np.array(row['geometry'].coords[0]) / scale_factor
        gaus = makeGaussian(output_size, fwhm=3, center=center)
        conv_pts.append(gaus)
    conv_pts.append(np.zeros((output_size, output_size)))
    conv_pts = np.array(conv_pts)
    return conv_pts


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def pad(image, padding_w, padding_h):
    batch_size, height, width, depth = image.shape
    # @TODO: Avoid creating new array
    new_image = np.zeros((batch_size, height + padding_h * 2, width + padding_w * 2, depth), dtype=image.dtype)
    new_image[:, padding_h:(height + padding_h), padding_w:(width + padding_w)] = image
    # @TODO: Fill padded zones
    # new_image[:, :padding_w] = image[:, :padding_w]
    # new_image[:padding_h, :] = image[:padding_h, :]
    # new_image[-padding_h:, :] = image[-padding_h:, :]

    return new_image


def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled


def unpad(image, padding_w):
    return image[:, :, padding_w:(image.shape[1] - padding_w), :]


def build_batch_generator_angle(filenames, img_man_dir=None, img_auto_dir=None, batch_size=None,
                          shuffle=False, transformations=None,
                          out_size=None, crop_size=None, mask_dir=None, aug=False, r_type='rgg'):
    mask_function = ImageWithMaskFunction(out_size=out_size, crop_size=crop_size, mask_dir=mask_dir)

    while True:
        # @TODO: Should we fixate the seed here?
        if shuffle:
            filenames = sklearn.utils.shuffle(filenames)

        for start in range(0, len(filenames), batch_size):
            batch_x = []
            masks = []
            weights = []
            end = min(start + batch_size, len(filenames))
            train_batch = filenames[start:end]

            for ind, filename in train_batch.iterrows():
                #img_path = os.path.join(img_man_dir, str(filename['folder']))
                img_load_path = os.path.join(img_man_dir, str(filename['folder']), 'images', filename['name'])
                img = img_to_array(
                    load_img(img_load_path, grayscale=False).resize(out_size))
                mask_load_path = os.path.join(img_man_dir, str(filename['folder']), 'np_arrays', filename['name'].replace(".png", ".npy"))
                mask = np.load(mask_load_path)
                batch_x.append(img)
                masks.append(mask)
                weights.append(filename['weight'])
            batch_x = np.array(batch_x, np.float32)
            masks = np.array(masks, np.float32)
            #batch_x, masks, angles = mask_function.mask_pred(batch_x, train_batch, range(batch_size), img_man_dir, img_auto_dir, aug, r_type)
            weights = np.array(weights)
            yield imagenet_utils.preprocess_input(batch_x, mode=args.preprocessing_function), masks #[masks, angles]


def get_edges(image):
    img = np.uint8(image[:,:,0])
    out = np.zeros((args.img_height, args.img_width, 5))
    laplacian = min_max(cv2.Laplacian(img, cv2.CV_64F))
    sobelx = min_max(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
    sobely = min_max(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))
    edges = min_max(cv2.Canny(img, 30, 200))
    for n, c in zip(range(5), [image[:, :, 0]/255, laplacian, sobelx, sobely, edges]):
        out[:, :, n] = c
    return out


def min_max(X):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled

