import os

import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# from CyclicLearningRate import CyclicLR
from research_code.datasets_tf2 import \
    DataGenerator_angles  # build_batch_generator, generate_filenames, build_batch_generator_angle
from research_code.losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border, mse_masked, angle_rmse
from research_code.models import make_model
from research_code.params import args
from research_code.utils import freeze_model, ThreadsafeIter

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def setup_env():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    setup_env()
    train_dir = args.manual_dataset_dir
    val_dir = args.test_data_dir

    if args.net_alias is not None:
        formatted_net_alias = '-{}-'.format(args.net_alias)

    os.makedirs("{}/{}".format(args.models_dir, args.exp_name), exist_ok=True)
    best_model_file = \
        '{}/{}/{}{}loss-{}-fold_{}-{}{:.6f}-{}'.format(args.models_dir, args.exp_name, args.network,
                                                       formatted_net_alias, args.loss_function,
                                                       args.fold, args.input_width,
                                                       args.learning_rate, args.r_type) + \
        '-{epoch:d}-{val_loss:0.7f}.h5'
    ch = 3
    # model = make_model((args.input_width, args.input_height, args.stacked_channels + ch))
    model = make_model((None, None, args.stacked_channels + ch))
    freeze_model(model, args.freeze_till_layer)
    # class_names = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster']
    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()
    num_classes = len(args.class_names)
    loss_list = [make_loss('bce_dice') for i in range(num_classes)]
    metrics_list = [dice_coef for i in range(num_classes)]
    model.compile(loss=loss_list,
                  optimizer=optimizer,
                  metrics=metrics_list)

    crop_size = None

    if args.use_crop:
        crop_size = (args.input_height, args.input_width)
        print('Using crops of shape ({}, {})'.format(args.input_height, args.input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')

    df = pd.read_csv(args.train_df)
    df = df[df[args.class_names[0]] != 0]
    train_df = df[df['ds_part'] == 'train']
    val_df = df[df['ds_part'] == 'val']

    print('Training fold #{}, {} in train_ids, {} in val_ids'.format(args.fold, len(train_df), len(val_df)))

    train_generator = DataGenerator_angles(
        train_df,
        classes=args.class_names,
        img_dir=train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        # mask_dir=mask_dir,
        aug=args.use_aug
    )

    val_generator = DataGenerator_angles(
        val_df,
        classes=args.class_names,
        img_dir=val_dir,
        batch_size=args.batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        # mask_dir=mask_dir,
        aug=False
    )

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')
    log_dir = os.path.join('./logs', args.exp_name)
    os.makedirs(log_dir, exist_ok=False)
    callbacks = [best_model,
                 EarlyStopping(patience=10, verbose=10),
                 TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                             write_images=True)]
    # ReduceLROnPlateau(monitor='val_dice_coef', mode='max', factor=0.2, patience=5, min_lr=0.00001,
    #                  verbose=1)]
    if args.clr is not None:
        clr_params = args.clr.split(',')
        base_lr = float(clr_params[0])
        max_lr = float(clr_params[1])
        step = int(clr_params[2])
        mode = clr_params[3]
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step, mode=mode)
        callbacks.append(clr)
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_df) / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_df) / args.batch_size + 1,
        callbacks=callbacks,
        max_queue_size=50,
        workers=4)


if __name__ == '__main__':
    main()
