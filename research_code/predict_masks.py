import os

import numpy as np
from keras.applications import imagenet_utils
import pandas as pd
import cv2
from research_code.models import make_model
from research_code.params import args
from research_code.evaluate import iou
from tqdm import tqdm
tqdm.monitor_interval = 0
from collections import OrderedDict

from research_code.random_transform_mask import pad_size, unpad

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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


def get_new_shape(img_shape, max_shape=224):
    if img_shape[0] > img_shape[1]:
        new_shape = (max_shape, int(img_shape[1] * max_shape / img_shape[0]))
    else:
        new_shape = (int(img_shape[0] * max_shape / img_shape[1]), max_shape)
    return new_shape


def predict_and_evaluate(thresh):
    output_dir = args.pred_mask_dir
    class_names = args.class_names
    os.makedirs(output_dir, exist_ok=True)
    model = make_model((None, None, 3 + args.stacked_channels))
    model.load_weights(args.weights)
    batch_size = 1
    test_df = pd.read_csv(args.test_df)
    nbr_test_samples = len(test_df)

    pbar = tqdm()
    class_ious = {cls: [] for cls in class_names}
    for i in range(int(nbr_test_samples / batch_size)):
        img_sizes = []

        for j in range(batch_size):
            if i * batch_size + j < len(test_df):
                class_labels = {}

                row = test_df.iloc[i * batch_size + j]
                img_path = os.path.join(
                    args.test_data_dir, 'images', "rgb", row['name'])
                # print(img_path)
                for cls in class_names:
                    mask_path = os.path.join(args.test_data_dir, 'labels', cls, row['name'])
                    mask = cv2.imread(mask_path.replace(".jpg", ".png"), cv2.IMREAD_GRAYSCALE)
                    # print(mask.shape)
                    class_labels[cls] = mask
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # mask = Image.open(mask_path)
                img_size = img.size
                # new_shape = get_new_shape(img_size, max_shape=512)
                # img = cv2.resize(img_to_array(img), new_shape)

                img, pads = pad_size(img)
                img_sizes.append(img_size)
                #masks.append(img_to_array(mask))
        x = np.expand_dims(img, axis=0)
        x = imagenet_utils.preprocess_input(x,  'channels_last', mode='tf')
        batch_x = x
        preds = model.predict_on_batch(batch_x)

        for num, pred in enumerate(preds):
            # predicted_masks.append((pred[0, :, :, 0] * 255).astype(np.uint8))
            bin_mask = (pred[0, :, :, 0] * 255).astype(np.uint8)
            cur_class = class_names[num]
            gt_mask = class_labels.get(cur_class)
            cur_iou = iou(gt_mask > 128, bin_mask > 128)
            class_ious[cur_class].append(cur_iou)
            row = test_df.iloc[i * batch_size]
            filename = row['name']
            save_folder_masks = os.path.join(output_dir, "labels", cur_class)
            os.makedirs(save_folder_masks, exist_ok=True)
            save_path_masks = os.path.join(save_folder_masks, filename)
            cv2.imwrite(save_path_masks, bin_mask)

        cur_iou_state = []
        for cls, ious in class_ious.items():
            cur_iou_state.append((cls, np.mean(ious)))
            # print("{}: {}".format(cls, np.)mean(ious)))
        iou_dict = OrderedDict(cur_iou_state)
        pbar.set_postfix(iou_dict)
        pbar.update(1)


if __name__ == '__main__':
    predict_and_evaluate(args.threshold)
