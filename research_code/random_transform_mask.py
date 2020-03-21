import random
import os

import keras.backend as K
import numpy as np
#from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, random_channel_shift, flip_axis, \
from keras.preprocessing.image import    load_img, img_to_array
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RGBShift, Rotate, Resize
)
import cv2

class ImageWithMaskFunction:
    def __init__(self, out_size, mask_dir, mask_suffix=".png", crop_size=None):
        super().__init__()
        self.out_size = out_size
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.crop_size = crop_size

    def prepare_lines(self, bin_mask, mask):
        mask = mask > 200
        line_tree_mask = mask.copy()
        bin_mask = bin_mask > 0
        line_tree_mask = line_tree_mask * bin_mask
        line_gap_mask = mask * np.bitwise_not(bin_mask)
        return line_tree_mask, line_gap_mask, mask


    def mask_pred(self, batch_x, filenames, img_auto_dir, aug=False):
        mask_pred = np.zeros((len(batch_x), self.out_size[0], self.out_size[1], 1), dtype=K.floatx())
        mask_pred[:, :, :, :] = 0.
        angle_pred = np.zeros((len(batch_x), self.out_size[0], self.out_size[1], 1), dtype=K.floatx())
        angle_pred[:, :, :, :] = 0.
        border_pred = angle_pred.copy()
        augmentation = strong_aug(p=0.9)
        for i, (ind, j) in enumerate(filenames.iterrows()):
            fname = j['name']
            mask = os.path.join(img_auto_dir, str(j['folder']), 'masks', fname)
            edge = os.path.join(img_auto_dir, str(j['folder']), 'edges', fname)
            bin_mask = os.path.join(img_auto_dir, str(j['folder']), 'bin_masks', fname)
            # mask_img = load_img(mask, grayscale=True, target_size=(self.out_size[0], self.out_size[1]))
            try:
                mask_img = img_to_array(load_img(mask, grayscale=True).resize(self.out_size))
                #edge_img = img_to_array(load_img(edge, grayscale=True).resize(self.out_size))
                if os.path.exists(bin_mask):
                    bin_mask_img = img_to_array(load_img(bin_mask, grayscale=True).resize(self.out_size))
                else:
                    bin_mask_img = np.zeros(mask_img.shape)
            except:
                print('hi')
                print(mask)
                raise ValueError(mask)
            if mask_img.shape[:2] != self.out_size:
                _, mask_img = pad_img(None, mask_img, self.out_size)
            line_tree_mask, line_gap_mask, mask = self.prepare_lines(bin_mask_img, mask_img)
            kernel = np.ones((5, 5))
            mask_pred[i, :, :, :] = line_tree_mask
            angle_pred[i, :, :, :] = line_gap_mask
            tree_dil = cv2.dilate(line_tree_mask.astype(np.uint8), kernel, iterations=1)
            gap_dil = cv2.dilate(line_gap_mask.astype(np.uint8), kernel, iterations=1)
            intersection = tree_dil * gap_dil
            border_pred[i, :, :, :] = np.expand_dims(intersection > 0, axis=-1)
            # mask_pred[i, :, :, :] = mask_img / 255
            # angle_pred[i, :, :, :] = (mask_img > 1) & (200 > mask_img)
            # border_pred[i, :, :, :] = mask_img < 200
            # border_pred[i, :, :, :] = mask_img * (np.pi / 180)
            #edge_img[mask_img > 0] = 0
            #angle_pred[i, :, :, :] = edge_img > 0
            #
            if aug:
                data = {"image": batch_x[i, :, :, :].astype(np.uint8),
                        'mask': mask_pred[i, :, :, :].astype(np.float32),
                        'angle': angle_pred[i, :, :, :].astype(np.float32),
                        'border': border_pred[i, :, :, :].astype(np.float32)}
                augmented = augmentation(**data)
                try:
                    batch_x[i, :, :, :], mask_pred[i, :, :, :], angle_pred[i, :, :, :], border_pred[i, :, :, :] = augmented["image"],\
                                                                 augmented['mask'].reshape((augmented['mask'].shape[0],
                                                                                                                augmented['mask'].shape[1],
                                                                                                                1)),\
                                                                 augmented['angle'].reshape((augmented['angle'].shape[0],
                                                                                           augmented['angle'].shape[1],
                                                                                           1)),\
                                                                    augmented['border'].reshape((augmented['border'].shape[0],
                                                                                                augmented['border'].shape[1],
                                                                                                1))
                except:
                    print('hi')

        if self.crop_size:
            height = self.crop_size[0]
            width = self.crop_size[1]
            ori_height = self.out_size[0]
            ori_width = self.out_size[1]
            if aug:
                h_start = random.randint(0, ori_height - height - 1)
                w_start = random.randint(0, ori_width - width - 1)
            else:
                # validate on center crops
                h_start = (ori_height - height) // 2
                w_start = (ori_width - width) // 2
            MASK_CROP = mask_pred[:, h_start:h_start + height, w_start:w_start + width, :]
            ANGLE_CROP = angle_pred[:, h_start:h_start + height, w_start:w_start + width, :]
            BORDER_CROP = angle_pred[:, h_start:h_start + height, w_start:w_start + width, :]
            return batch_x[:, h_start:h_start + height, w_start:w_start + width, :], MASK_CROP, ANGLE_CROP, BORDER_CROP
        else:
            return batch_x, mask_pred

    def mask_pred_angles(self, batch_x, masks_x, classes, aug=False):
        augmentation = strong_aug(p=0.6)
        for i in range(len(batch_x)):
            if aug:
                data = {"image": batch_x[i, :, :, :].astype(np.uint8)}
                for cls in classes:
                    data[cls] = masks_x[cls][i, :, :, :].astype(np.float32)
                augmented = augmentation(**data)
                batch_x[i, :, :, :] = augmented["image"]
                for cls in classes:
                    masks_x[cls][i, :, :, :] = augmented[cls].reshape((augmented[cls].shape[0],
                                                                       augmented[cls].shape[1],
                                                                       augmented[cls].shape[2]))
        # print(len(masks_x.keys()))
        if self.crop_size:
            mask_crop = {}
            height = self.crop_size[0]
            width = self.crop_size[1]
            ori_height = self.out_size[0]
            ori_width = self.out_size[1]
            if aug:
                h_start = random.randint(0, ori_height - height - 1)
                w_start = random.randint(0, ori_width - width - 1)
            else:
                # validate on center crops
                h_start = (ori_height - height) // 2
                w_start = (ori_width - width) // 2
            img_crop = batch_x[:, h_start:h_start + height, w_start:w_start + width, :]
            for cls in classes:
                print(masks_x[cls])
                mask_crop[cls] = masks_x[cls][:, h_start:h_start + height, w_start:w_start + width, :]
            return img_crop, mask_crop
        else:
            return batch_x, masks_x


    def mask_pred_train(self, batch_x, filenames, index_array, l):
        return self.mask_pred(batch_x, filenames, index_array, True)

    def mask_pred_val(self, batch_x, filenames, index_array, l):
        return self.mask_pred(batch_x, filenames, index_array, False)


def get_window(mask, attitude=120, window_meters=30, focal=35.4):
    # camera params
    #focal = 35.4
    width = mask.shape[0]
    height = mask.shape[1]
    sensor_width = 35.9
    meters = window_meters

    # calcluate field of view in meters (base image atitude used, because We will scale target image to base and translate it
    fov = sensor_width * attitude / focal
    #print(fov)

    # calcluate number of pixels in 1 meter
    pix_in_meter = width / fov
    #print(pix_in_meter)

    # transform the difference from meters to pixels
    x_dif_pix = meters * pix_in_meter
#     y_dif_pix = meters * pix_in_meter
    return int(x_dif_pix)


def get_bigger_window(step_row, row, step_col, col, window_size_pixels, height, width):
    pad_size = [0, 0, 0, 0]
    new_step_row = int(step_row - window_size_pixels // 2)
    if new_step_row < 0:
        pad_size[0] = -new_step_row
    new_row = int(row + window_size_pixels // 2)
    if new_row > height:
        pad_size[1] = new_row - height
    new_col = int(col - window_size_pixels // 2)
    if new_col < 0:
        pad_size[2] = -new_col
    new_step_col = int(step_col + window_size_pixels // 2)
    if new_step_col > width:
        pad_size[3] = new_step_col - width
    new_window = (new_step_row, new_row, new_col, new_step_col)

    return new_window, pad_size


def pad_img(img, mask, shape):
    # pad_shape = np.int8((np.array(shape) - np.array(mask.shape[:2]))/2)
    padded_img = None
    padded_mask = None
    # print(pad_shape)
    if isinstance(mask, np.ndarray):
        pad_shape = np.int16(np.ceil(((np.array(shape) - np.array(mask.shape[:2])) / 2)))
        if pad_shape.min() < 0:
            padded_mask = cv2.resize(mask, shape)
            padded_mask = np.expand_dims(padded_mask, axis=2)
        else:
            padded_mask = np.pad(mask, ((pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1]), (0, 0)), 'reflect')
            padded_mask = padded_mask[:shape[0], :shape[1], :]
    if isinstance(img, np.ndarray):
        pad_shape = np.int16(np.ceil((np.array(shape) - np.array(img.shape[:2])) / 2))
        if pad_shape.min() < 0:
            padded_img = cv2.resize(img, shape)
        else:
            padded_img = np.zeros((img.shape[0] + 2 * pad_shape[0],
                                   img.shape[1] + 2 * pad_shape[1], 3), dtype=np.uint8)
            for i in range(3):
                padded_img[:, :, i] = np.pad(img[:, :, i], ((pad_shape[0],pad_shape[0]), (pad_shape[1],pad_shape[1])), 'reflect')
            padded_img = padded_img[:shape[0], :shape[1], :]
    # print(paded_mask.shape, paded_img.shape)
    return padded_img, padded_mask

def pad_size(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def pad(img, shape):#pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if shape == 0:
        return img
    pad_shape = np.int16(np.ceil((np.array(shape) - np.array(img.shape[:2]))))
    height, width = img.shape[:2]

    # if height % shape == 0:
    #     y_min_pad = 0
    #     y_max_pad = 0
    # else:
    y_pad = pad_shape[0]
    y_min_pad = int(y_pad / 2)
    y_max_pad = y_pad - y_min_pad

    # if width % pad_size == 0:
    #     x_min_pad = 0
    #     x_max_pad = 0
    # else:
    x_pad = pad_shape[1]
    x_min_pad = int(x_pad / 2)
    x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]

# height_shift_range=0.2,
# width_shift_range=0.2,
# shear_range=0.0,
# rotation_range=45,
# zoom_range=[0.7, 1.2],
# channel_shift_range=0.1,
# horizontal_flip=True,
# vertical_flip=True)
def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

def strong_aug_dist(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=0.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            # IAAEmboss(),
            # RandomContrast(),
            # RandomBrightness(),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def tiles_with_overlap(img, window_size, overlap):
    sp = []
    matrices = []
    pnt_step = int(window_size * overlap)
    step_h = img.shape[1]//pnt_step
    step_w = img.shape[0]//pnt_step
    # print(step_h, step_w)
    pointerh_min = 0
    for h in range(step_h + 1):
        if h != 0:
            pointerh_min += pnt_step
        pointerh = min(pointerh_min, img.shape[1])
        pointerh_max = pointerh_min + window_size
        pointerh_max = min(pointerh_min + window_size, img.shape[1])
        pointerw_min = 0
        if pointerh == pointerh_max:
                #print("hi")
                continue
        for w in range(step_w + 1):
            if w != 0:
                pointerw_min += pnt_step
            pointerw = min(pointerw_min, img.shape[0])
            pointerw_max = pointerw_min + window_size
            pointerw_max = min(pointerw_min + window_size, img.shape[0])
            if pointerw == pointerw_max:
                #print("hi")
                continue
            else:
                # print((pointerh, pointerh_max), (pointerw, pointerw_max))
                sp.append([pointerh, pointerh_max, pointerw, pointerw_max])
                # matrices.append(img[pointerh:pointerh_max, pointerw:pointerw_max])
                matrices.append(img[pointerw:pointerw_max, pointerh:pointerh_max])
                # print(img[pointerh:pointerh_max, pointerw:pointerw_max].shape)
    return matrices, sp


def tiles_with_overlap_shape(w_img, h_img, window_size, overlap):
    sp = []
    pnt_step = int(window_size * overlap)
    step_h = h_img // pnt_step
    step_w = w_img // pnt_step
    pointerh_min = 0
    for h in range(step_h + 1):
        if h != 0:
            pointerh_min += pnt_step
        pointerh = min(pointerh_min, h_img)
        pointerh_max = min(pointerh_min + window_size, h_img)
        pointerw_min = 0
        if pointerh == pointerh_max:
            continue
        for w in range(step_w + 1):
            if w != 0:
                pointerw_min += pnt_step
            pointerw = min(pointerw_min, w_img)
            pointerw_max = min(pointerw_min + window_size, w_img)
            if pointerw == pointerw_max:
                continue
            sp.append([pointerw, pointerw_max, pointerh, pointerh_max])
    return sp


def read_img_opencv(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb2rgg(rgb):
    rgg = rgb.copy()
    rgg[:, :, 2] = rgg[:, :, 1]
    # rgg[:, :, 0] = rgg[:, :, 0] / 5.87
    # rgg[:, :, 1] = rgg[:, :, 1] / 5.95
    # rgg[:, :, 2] = rgg[:, :, 2] / 5.95
    return rgg


def create_nrg(path):
    red_path = path.replace(".JPG", "_Red.JPG")
    green_path = path.replace(".JPG", "_Green.JPG")
    nir_path = path.replace(".JPG", "_NIR.JPG")
    nrg_path = nir_path.replace("_NIR.JPG", "_NRG.JPG")
    if os.path.exists(nrg_path):
        print("{} exists".format(nrg_path))
        return nrg_path
    red_img = cv2.imread(red_path, 0)
    green_img = cv2.imread(green_path, 0)
    nir_img = cv2.imread(nir_path, 0)
    # nrg_path = nir_path.replace("_NIR.JPG", "_NRG.JPG")
    # DSC06633_Blue.JPG
    grn = np.zeros((nir_img.shape[0], nir_img.shape[1], 3))
    grn[:, :, 0] = green_img
    grn[:, :, 1] = red_img
    grn[:, :, 2] = nir_img
    cv2.imwrite(nrg_path, grn)
    return nrg_path