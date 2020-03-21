import os
from time import clock

import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img, flip_axis
from PIL import Image
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import cv2

from models import make_model
from params import args
from evaluate import dice_coef, calculate_metrics
from datasets import generate_images
from tqdm import tqdm
tqdm.monitor_interval = 0

from random_transform_mask import pad_size, unpad

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
prediction_dir = args.pred_mask_dir


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

# def predict():
#     output_dir = args.pred_mask_dir
#     model = make_model((None, None, 3))
#     model.load_weights(args.weights)
#     batch_size = args.pred_batch_size
#     nbr_test_samples = 100064
#
#     filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]
#
#     start_time = clock()
#     for i in range(int(nbr_test_samples / batch_size) + 1):
#         x = []
#         for j in range(batch_size):
#             if i * batch_size + j < len(filenames):
#                 img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
#                 x.append(img_to_array(img))
#         x = np.array(x)
#         x = preprocess_input(x, args.preprocessing_function)
#         x = do_tta(x, args.pred_tta)
#         batch_x = np.zeros((x.shape[0], 1280, 1920, 3))
#         batch_x[:, :, 1:-1, :] = x
#         preds = model.predict_on_batch(batch_x)
#         preds = undo_tta(preds, args.pred_tta)
#         for j in range(batch_size):
#             filename = filenames[i * batch_size + j]
#             prediction = preds[j][:, 1:-1, :]
#             array_to_img(prediction * 255).save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
#         time_spent = clock() - start_time
#         print("predicted batch ", str(i))
#         print("Time spent: {:.2f}  seconds".format(time_spent))
#         print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
#         print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))


def predict():
    output_dir = args.pred_mask_dir
    model = make_model((None, None, 3))
    model.load_weights(args.weights)
    batch_size = args.pred_batch_size
    thresh = args.threshold
    test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    nbr_test_samples = len(test_df)
    filenames = [os.path.join(args.test_data_dir, f) for f in sorted(os.listdir(args.test_data_dir))]

    start_time = clock()
    for i in range(int(nbr_test_samples / batch_size)):
        x = []
        img_sizes = []
        for j in range(batch_size):
            if i * batch_size + j < len(filenames):
                # img = imread(os.path.join(img_dir, filename))
                # img = load_img(filenames[i * batch_size + j], target_size=(args.img_height, args.img_width))
                img = Image.open(filenames[i * batch_size + j])
                img_size = img.size
                img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                img_sizes.append(img_size)
                if args.edges:
                    img = generate_images(img)

                x.append(img_to_array(img))
        x = np.array(x)
        x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        # preds = undo_tta(preds, args.pred_tta)
        for j in range(batch_size):
            filename = filenames[i * batch_size + j]
            print(filename)
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            prediction = prediction > thresh
            pred_im = array_to_img(prediction * 255).resize(img_sizes[j], Image.ANTIALIAS)
            try:
                assert pred_im.size == img_size
            except:
                print('bad')
            pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        time_spent = clock() - start_time
        print("predicted batch ", str(i))
        print("Time spent: {:.2f}  seconds".format(time_spent))
        print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
        print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))


def predict_and_evaluate():
    r_type = args.r_type
    output_dir = args.pred_mask_dir
    thresh = args.threshold
    os.makedirs(output_dir, exist_ok=True)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))
    # test_mask_dir = args.test_mask_dir
    model = make_model((None, None, 3 + args.stacked_channels))
    model.load_weights(args.weights)
    batch_size = 1
    test_df = pd.read_csv(args.test_df)
    nbr_test_samples = len(test_df)
    dices = []
    ious = []
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        img_sizes = []
        x = []
        masks = []
        for j in range(batch_size):
            if i * batch_size + j < len(test_df):
                stacked_channels= []
                row = test_df.iloc[i * batch_size + j]
                if 'src_name' in row.index.tolist():
                    img_path = os.path.join(args.test_data_dir, row['folder'], r_type, row['src_name'])
                else:
                    img_path = os.path.join(args.test_data_dir, row['folder'], r_type, row['name'].replace('.png', '.jpg').replace('nrg', r_type).replace('rgg', r_type))
                mask_path = os.path.join(args.test_data_dir, row['folder'], 'nrg_masks', row['name'])
                img = Image.open(img_path)
                mask = Image.open(mask_path)
                img_size = img.size
                try:
                    img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                except:
                    print('hi')
                mask = mask.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                # mask = img_to_array(mask) > 0
                img_sizes.append(img_size)
                img = img_to_array(img)
                #stacked_channels.append(img)
                for k in range(args.stacked_channels):
                    if row['folder'] == "31_agrowing_1b1146":
                        channel_path = os.path.join(args.test_data_dir, row['folder'], args.stacked_channels_dir,
                                                    row['name'].replace('green', "rgbn").replace('nrg', "blue").replace('rgg',
                                                                                                    "blue").replace(
                                                        "project_transparent_reflectance_group1", "blue").replace(
                                                        '.png', '.jpg'))
                    else:
                        channel_path = os.path.join(args.test_data_dir, row['folder'], args.stacked_channels_dir,
                                                    row['name'].replace('nrg', "blue").replace('rgg', "blue").replace("project_transparent_reflectance_group1", "blue").replace(
                                                        '.png', '.jpg'))
                    stacked_channel = cv2.imread(channel_path, 0)
                    stacked_channel = cv2.resize(stacked_channel, (args.input_height, args.input_width))
                    stacked_channels.append(stacked_channel)
                stacked_img = np.dstack((img, *stacked_channels))

                # x.append(img)
                masks.append(img_to_array(mask) > 128)
        x = np.expand_dims(stacked_img, axis=0)
        masks = np.array(masks)
        if not args.edges:
             x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        # preds = undo_tta(preds, args.pred_tta)
        batch_iou, batch_dice = calculate_metrics(masks, preds > thresh)
        # batch_iou = iou(masks, preds > thresh)
        dices.append(batch_dice)
        ious.append(batch_iou)
        for j in range(batch_size):
            row = test_df.iloc[i * batch_size + j]
            # filename = filenames[i * batch_size + j]
            filename = row['name']
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            prediction = prediction > thresh
            pred_im = array_to_img(prediction * 255).resize(img_sizes[j], Image.ANTIALIAS)
            if pred_im.size != img_sizes[j]:
                print("why")
            pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))

    print(np.mean(dices))
    print(np.mean(ious))

def predict_full():
    r_type = args.r_type
    output_dir = args.pred_mask_dir
    thresh = args.threshold
    os.makedirs(output_dir, exist_ok=True)
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # set_session(tf.Session(config=config))
    # test_mask_dir = args.test_mask_dir
    model = make_model((None, None, 3 + args.stacked_channels))
    model.load_weights(args.weights)
    batch_size = 1
    test_df = pd.read_csv(args.test_df)
    # nbr_test_samples = len(os.listdir(args.test_data_dir))
    nbr_test_samples = len(test_df)

    # filenames = [os.path.join(args.test_data_dir, f) for f in test_df['name']]
    # mask_filenames = [os.path.join(args.test_mask_dir, f).replace('.jpg', '.png') for f in test_df['name']]
    # filenames = [f for f in test_df['name']]
    # mask_filenames = [f.replace('.jpg', '.png') for f in test_df['name']]

    start_time = clock()
    dices = []
    pbar = tqdm(total=nbr_test_samples // batch_size)
    # for i in range(10):
    #     time.sleep(0.1)
    #     pbar.update(10)

    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        img_sizes = []
        x = []
        pads = []
        masks = []
        for j in range(batch_size):
            if i * batch_size + j < len(test_df):
                stacked_channels = []
                row = test_df.iloc[i * batch_size + j]
                img_path = os.path.join(args.test_data_dir, row['folder'], r_type + "_labeled", row['name'].replace('.png', '.jpg').replace('nrg', r_type))
                mask_path = os.path.join(args.test_data_dir, row['folder'], 'nrg_masks', row['name'])
                try:
                    img = Image.open(img_path)
                except:
                    print('Hi')
                mask = Image.open(mask_path)
                img_size = img.size
                # img = img.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                # mask = mask.resize((args.input_height, args.input_width), Image.ANTIALIAS)
                # mask = img_to_array(mask) > 0
                img_sizes.append(img_size)
                if args.edges:
                    img = generate_images(img_to_array(img))
                else:
                    img = img_to_array(img)
                img, _pad = pad_size(img)
                pads.append(_pad)
                for k in range(args.stacked_channels):
                    if row['folder'] == "31_agrowing_1b1146":
                        channel_path = os.path.join(args.test_data_dir, row['folder'], args.stacked_channels_dir,
                                                    row['name'].replace('green', "rgbn").replace('nrg', "blue").replace('rgg',
                                                                                                    "blue").replace(
                                                        "project_transparent_reflectance_group1", "blue").replace(
                                                        '.png', '.jpg'))
                    else:
                        channel_path = os.path.join(args.test_data_dir, row['folder'], args.stacked_channels_dir,
                                                    row['name'].replace('nrg', "blue").replace('rgg', "blue").replace("project_transparent_reflectance_group1", "blue").replace(
                                                        '.png', '.jpg'))
                    stacked_channel = cv2.imread(channel_path, 0)
                    stacked_channel, _bluepad = pad_size(stacked_channel)
                    stacked_channels.append(stacked_channel)
                stacked_img = np.dstack((img, *stacked_channels))
                #x.append(img)
                masks.append(img_to_array(mask) > 128)
        x = np.array(stacked_img)
        x = np.expand_dims(x, axis=0)
        masks = np.array(masks)
        if not args.edges:
             x = imagenet_utils.preprocess_input(x, mode=args.preprocessing_function)
        # x = imagenet_utils.preprocess_input(x, args.preprocessing_function)
        # x = do_tta(x, args.pred_tta)
        batch_x = x
        # batch_x = np.zeros((x.shape[0], 887, 887, 3))
        # batch_x[:, :, 1:-1, :] = x
        preds = model.predict_on_batch(batch_x)
        # preds = undo_tta(preds, args.pred_tta)

        for j in range(batch_size):
            row = test_df.iloc[i * batch_size + j]
            # filename = filenames[i * batch_size + j]
            filename = row['name']
            # prediction = preds[j][:, 1:-1, :]
            prediction = preds[j]
            prediction = prediction > thresh
            prediction = unpad(prediction, pads[j])
            try:
                dice = dice_coef(masks[j], prediction)
                dices.append(dice)
                # print("ok")
            except:
                print(masks[j].shape, prediction.shape, x[j].shape)
            pred_im = array_to_img(prediction * 255)#.resize(img_sizes[j], Image.ANTIALIAS)
            pred_im.save(os.path.join(output_dir, filename.split('/')[-1][:-4] + ".png"))
        pbar.set_postfix(Dice='{}'.format(np.mean(dices)))
        pbar.update(1)
    #    time_spent = clock() - start_time
    #     print("predicted batch ", str(i))
    #     print("Time spent: {:.2f}  seconds".format(time_spent))
    #     print("Speed: {:.2f}  ms per image".format(time_spent / (batch_size * (i + 1)) * 1000))
    #     print("Elapsed: {:.2f} hours  ".format(time_spent / (batch_size * (i + 1)) / 3600 * (nbr_test_samples - (batch_size * (i + 1)))))
    #     print("predicted batch dice {}".format(batch_dice))
    pbar.close()
    print(np.mean(dices))


if __name__ == '__main__':
    # predict()
    predict_and_evaluate()
    # predict_full()
