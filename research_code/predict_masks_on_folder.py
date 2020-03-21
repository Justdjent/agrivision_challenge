import os
from time import clock

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from models import make_model
from params import args
from evaluate import dice_coef
from datasets import generate_images
from tqdm import tqdm
import cv2
from random_transform_mask import pad, unpad, tiles_with_overlap, read_img_opencv, rgb2rgg, get_window, create_nrg
from datasets import build_batch_generator_predict_folder

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
prediction_dir = args.pred_mask_dir


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def do_tta(x, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(x, 2)
    else:
        return x


def undo_tta(pred, tta_type):
    if tta_type == 'hflip':
        # batch, img_col = 2
        return flip_axis(pred, 2)
    else:
        return pred




def predict_folder():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    #tile_size = 2652
    overlap = 0.5
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    set_session(tf.Session(config=config))
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = 1
    thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir)) if f.endswith(".JPG")]
    nbr_test_samples = len(filenames)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img_path = os.path.join(args.test_data_dir, filenames[i * batch_size + j])
                # img = Image.open(os.path.join(args.test_data_dir, filenames[i * batch_size + j]))
                img = read_img_opencv(img_path)
                tile_size = get_window(img, attitude=40, window_meters=30)

                img_size = img.shape
                # img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)

                # img = rgb2rgg(img)
                img_sizes.append(img_size)
                # tls, rects = split_on_tiles(img, overlap=1)
                # tls, rects = split_mask(img)
                tls, rects = tiles_with_overlap(img, tile_size, overlap)
                tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                for tile in tls:
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        padded_tile, pad_ = pad(tile, tile_size)
                        padded_tiles.append(padded_tile)
                        pads.append(pad_)
                    else:
                        padded_tiles.append(tile)
                        pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in padded_tiles]
                #tls = [img_to_array(tile) for tile in padded_tiles]

                #tls = padded_tiles
                # tile_sizes = [tile.shape for tile in tls]
                # x.append(tls)
        x = np.array(tls)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        num_x = x.shape[0]
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        if args.pred_tta:
            x_tta = do_tta(x, "hflip")
            batch_x = np.concatenate((x, x_tta), axis=0)
        else:
            batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds_net = model.predict_on_batch(batch_x)
        if args.pred_tta:
            preds_tta = undo_tta(preds_net[num_x:], "hflip")
            preds = (preds_net[:num_x] + preds_tta) / 2
        else:
            preds = preds_net
        pred = np.zeros((img.shape[0], img.shape[1]))
        for r, p, s, pad_ in zip(rects, preds, tile_sizes, pads):
            try:
                # res_pred = cv2.resize(p * 255, (s[1], s[0]))
                res_pred = cv2.resize(p * 255, (tile_size, tile_size))
                # res_pred = p * 255
                res_pred = unpad(res_pred, pad_)
                # stack_arr = np.dstack([res_pred, pred[r[1][0]:r[1][1], r[0][0]:r[0][1]]])
                stack_arr = np.dstack([res_pred, pred[r[2]:r[3], r[0]:r[1]]])
                pred[r[2]:r[3], r[0]:r[1]] = np.amax(stack_arr, axis=2)
                # pred[r[1]:r[1] + s[0], r[0]:r[0] + s[1]] = np.mean(stack_arr, axis=2)
            except:
                print('hi')
        #prediction = pred > thresh * 255
        prediction = pred * 255
        # preds = undo_tta(preds, args.pred_tta)
        # for j in range(batch_size):
        filename = filenames[i * batch_size + j]
        # print(filename)
        # prediction = preds[j][:, 1:-1, :]
        # prediction = preds[j]
        # prediction = prediction > thresh
        pred_im = array_to_img(np.expand_dims(prediction, axis=2))
        try:
            assert pred_im.size == (img_size[1], img_size[0])
        except:
            print('bad')
        pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))


def predict_folder_nrg():
    output_dir = args.pred_mask_dir
    os.makedirs(output_dir, exist_ok=True)
    #tile_size = 2652
    overlap = 0.5
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # set_session(tf.Session(config=config))
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = 1
    thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # self.loc['Name'] = [name.split('_')[0].replace(".JPG", "") for name in self.loc['imageName']]
    # self.loc = self.loc[~self.loc['Name'].duplicated()]
    # nbr_test_samples = len(os.listdir(args.test_data_dir))

    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir)) if f.endswith(".JPG")]
    names = [name.split('/')[-1].split("_")[0].replace(".JPG", "") for name in filenames]
    names = set(names)
    filenames = [os.path.join(args.test_data_dir, f + ".JPG") for f in list(names)]
    nbr_test_samples = len(filenames)
    start_time = clock()
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                img_path = filenames[i * batch_size + j]
                img_path = create_nrg(img_path)
                # img = Image.open(os.path.join(args.test_data_dir, filenames[i * batch_size + j]))
                img = read_img_opencv(img_path)
                tile_size = get_window(img, attitude=40, window_meters=30, focal=55)

                img_size = img.shape
                # img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)

                img_sizes.append(img_size)
                # tls, rects = split_on_tiles(img, overlap=1)
                # tls, rects = split_mask(img)
                tls, rects = tiles_with_overlap(img, tile_size, overlap)
                tile_sizes = [tile.shape for tile in tls]
                padded_tiles = []
                pads = []
                for tile in tls:
                    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                        padded_tile, pad_ = pad(tile, tile_size)
                        padded_tiles.append(padded_tile)
                        pads.append(pad_)
                    else:
                        padded_tiles.append(tile)
                        pads.append((0, 0, 0, 0))
                # tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in tls]
                tls = [img_to_array(cv2.resize(tile, (args.input_height, args.input_width))) for tile in padded_tiles]
                #tls = [img_to_array(tile) for tile in padded_tiles]

                #tls = padded_tiles
                # tile_sizes = [tile.shape for tile in tls]
                # x.append(tls)
        x = np.array(tls)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        num_x = x.shape[0]
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        if args.pred_tta:
            x_tta = do_tta(x, "hflip")
            batch_x = np.concatenate((x, x_tta), axis=0)
        else:
            batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds_net = model.predict_on_batch(batch_x)
        if args.pred_tta:
            preds_tta = undo_tta(preds_net[num_x:], "hflip")
            preds = (preds_net[:num_x] + preds_tta) / 2
        else:
            preds = preds_net
        pred = np.zeros((img.shape[0], img.shape[1]))
        for r, p, s, pad_ in zip(rects, preds, tile_sizes, pads):
            try:
                # res_pred = cv2.resize(p * 255, (s[1], s[0]))
                res_pred = cv2.resize(p * 255, (tile_size, tile_size))
                # res_pred = p * 255
                res_pred = unpad(res_pred, pad_)
                # stack_arr = np.dstack([res_pred, pred[r[1][0]:r[1][1], r[0][0]:r[0][1]]])
                stack_arr = np.dstack([res_pred, pred[r[2]:r[3], r[0]:r[1]]])
                pred[r[2]:r[3], r[0]:r[1]] = np.amax(stack_arr, axis=2)
                # pred[r[1]:r[1] + s[0], r[0]:r[0] + s[1]] = np.mean(stack_arr, axis=2)
            except:
                print('hi')
        #prediction = pred > thresh * 255
        prediction = pred * 255
        # preds = undo_tta(preds, args.pred_tta)
        # for j in range(batch_size):
        filename = filenames[i * batch_size + j]
        # print(filename)
        # prediction = preds[j][:, 1:-1, :]
        # prediction = preds[j]
        # prediction = prediction > thresh
        pred_im = array_to_img(np.expand_dims(prediction, axis=2))
        try:
            assert pred_im.size == (img_size[1], img_size[0])
        except:
            print('bad')
        pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))


def predict_folder_gen():
    output_dir = args.pred_mask_dir
    # tile_size = 2652
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=config))
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    # batch_size = 1
    # thresh = args.threshold
    # test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    overlap = 0.5
    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
    img_path = os.path.join(args.test_data_dir, filenames[0])
    img = read_img_opencv(img_path)
    tile_size = get_window(img, attitude=40, window_meters=30)

    img_size = img.shape
    tls, rects = tiles_with_overlap(img, tile_size, overlap)
    tile_sizes = [tile.shape for tile in tls]
    pred_gen = build_batch_generator_predict_folder(filenames=filenames, overlap=overlap, batch_size=None)
    nbr_test_samples = len(filenames)
    num_x = len(tls)
    padded_tiles = []
    pads = []
    for tile in tls:
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            padded_tile, pad_ = pad(tile, tile_size)
            padded_tiles.append(padded_tile)
            pads.append(pad_)
        else:
            padded_tiles.append(tile)
            pads.append((0, 0, 0, 0))
    #start_time = clock()
    # for i in tqdm(range(int(nbr_test_samples / batch_size))):
    preds_net = model.predict_generator(pred_gen,
                                        steps=nbr_test_samples,
                                        max_queue_size=50,
                                        workers=8)
    if args.pred_tta:
        preds_tta = undo_tta(preds_net[num_x:], "hflip")
        preds = (preds_net[:num_x] + preds_tta) / 2
    else:
        preds = preds_net
    pred = np.zeros((img.shape[0], img.shape[1]))
    for r, p, s, pad_ in zip(rects, preds, tile_sizes, pads):
        try:
            # res_pred = cv2.resize(p * 255, (s[1], s[0]))
            res_pred = cv2.resize(p * 255, (tile_size, tile_size))
            # res_pred = p * 255
            res_pred = unpad(res_pred, pad_)
            # stack_arr = np.dstack([res_pred, pred[r[1][0]:r[1][1], r[0][0]:r[0][1]]])
            stack_arr = np.dstack([res_pred, pred[r[2]:r[3], r[0]:r[1]]])
            pred[r[2]:r[3], r[0]:r[1]] = np.amax(stack_arr, axis=2)
            # pred[r[1]:r[1] + s[0], r[0]:r[0] + s[1]] = np.mean(stack_arr, axis=2)
        except:
            print('hi')
    # prediction = pred > thresh * 255
    prediction = pred * 255
    # preds = undo_tta(preds, args.pred_tta)
    # for j in range(batch_size):
    filename = filenames[i * batch_size + j]
    # print(filename)
    # prediction = preds[j][:, 1:-1, :]
    # prediction = preds[j]
    # prediction = prediction > thresh
    pred_im = array_to_img(np.expand_dims(prediction, axis=2))
    try:
        assert pred_im.size == (img_size[1], img_size[0])
    except:
        print('bad')
    pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
    print("hi")


def split_mask(mask, step=3536):
    """
    Splits mask on tiles
    :param step_h: height of tile
    :param step_w: width of tile
    :param image: image that needs to be splitted
    :return: list of tiles, starting point of each tile
    """
    step_h = step_w = step
    height, width, C = mask.shape
    overlap = 0.5
    matrices = []
    sp = []
    cnt_h = 0
    cnt_w = 0
    while height > 0:
        if height > step_h:
            b = (cnt_h + 1) * step_h
            a = cnt_h * step_h
        else:
            b = mask.shape[0]
            a = mask.shape[0] - step_h
        width = mask.shape[1]
        cnt_w = 0
        while width > 0:
            if width > step_w:
                zeros = mask[a:b, cnt_w * step_w:(cnt_w + 1) * step_w]
                sp.append((cnt_w * step_w, a))
            else:
                zeros = mask[a:b, (mask.shape[1] - step_w):mask.shape[1]]
                sp.append(((mask.shape[1] - step_w), a))
            cnt_w += 1#  - overlap
            matrices.append(zeros)

            width = width - step_w * overlap
        cnt_h += 1#  - overlap
        height = height - step_h * overlap
    return matrices, sp


def split_on_tiles(img, overlap=1, window_size_pixels=5304):
    window_size_pixels = window_size_pixels
    height, width, chan = img.shape
    print(height, width, chan)
    cnt = 0
    rects = []
    xs = 0
    tiles = []
    while xs < width:
        ys = 0
        while ys < height:
            step_row = xs + int(window_size_pixels)
            step_col = ys + int(window_size_pixels)

            rect = [[xs, min(width, step_row)], [ys, min(height, step_col)]]
            print(rect)
            tile = img[rect[0][0]:rect[0][1], rect[1][0]:rect[1][1]]
            tiles.append(tile)
            rects.append(rect)
            cnt += 1
            ys = ys + int(overlap * window_size_pixels)

        xs = min(xs + int(overlap * window_size_pixels), width)
    print(len(tiles))
    return tiles, rects



if __name__ == '__main__':
    # predict()
    # predict_and_evaluate()
    predict_folder()
    # predict_folder_nrg()
    # predict_folder_gen()
