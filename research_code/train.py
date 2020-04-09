import os

import pandas as pd
import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from research_code.data_generator import DataGeneratorSingleOutput
from research_code.losses import make_loss, dice_coef
from research_code.models import make_model
from research_code.params import args
from research_code.utils import freeze_model


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


def train():
    if args.exp_name is None:
        raise ValueError("Please add a name for your experiment - exp_name argument")
    setup_env()
    train_dir = args.train_dir
    val_dir = args.val_dir

    if args.net_alias is not None:
        formatted_net_alias = '-{}-'.format(args.net_alias)
    experiment_dir = os.path.join(args.experiments_dir, args.exp_name)
    model_dir = os.path.join(experiment_dir, args.models_dir)
    log_dir = os.path.join(experiment_dir, args.log_dir)
    if os.path.exists(log_dir) and len(os.listdir(log_dir)) > 0:
        raise ValueError("Please check if this experiment was already run (logs aren't empty) {}".format(experiment_dir))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_model_file = \
        '{}/{}{}loss-{}-{}{:.6f}'.format(model_dir, args.network,
                                         formatted_net_alias, args.loss_function, args.crop_width,
                                         args.learning_rate) + \
        '-{epoch:d}-{val_loss:0.7f}.h5'
    ch = 3
    activation = args.activation
    model = make_model((None, None, args.stacked_channels + ch),
                       network=args.network,
                       channels=len(args.class_names),
                       activation=activation)

    freeze_model(model, args.freeze_till_layer)
    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    optimizer = Adam(lr=args.learning_rate)

    if args.show_summary:
        model.summary()
    loss_list = [make_loss('bce_dice')]
    metrics_list = [dice_coef]
    model.compile(loss=loss_list,
                  optimizer=optimizer,
                  metrics=metrics_list)

    crop_size = None

    if args.use_crop:
        crop_size = (args.crop_height, args.crop_width)
        print('Using crops of shape ({}, {})'.format(args.crop_height, args.crop_width))
    else:
        print('Using full size images, --use_crop=True to do crops')
    dataset_df = pd.read_csv(args.dataset_df)

    train_df = dataset_df[dataset_df["ds_part"] == "train"]
    if args.exclude_bad_labels_df:
        invalid_df = pd.read_csv(args.exclude_bad_labels_df)
        train_df = pd.merge(train_df, invalid_df, on='name', how='outer')
        train_df['invalid'] = train_df['invalid'].fillna(False)
        train_df = train_df[~train_df['invalid']]
    val_df = dataset_df[dataset_df["ds_part"] == "val"]

    print('{} in train_ids, {} in val_ids'.format(len(train_df), len(val_df)))

    train_generator = DataGeneratorSingleOutput(
        train_df,
        classes=args.class_names,
        img_dir=train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=args.use_aug,
        validate_pixels=True,
        activation=activation
    )

    val_generator = DataGeneratorSingleOutput(
        val_df,
        classes=args.class_names,
        img_dir=val_dir,
        batch_size=args.batch_size,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=False,
        validate_pixels=True,
        activation=activation
    )

    best_model = ModelCheckpoint(best_model_file, monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')

    callbacks = [best_model,
                 EarlyStopping(patience=25, verbose=10),
                 TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                             write_images=True)]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_df) / args.batch_size + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_df) / args.batch_size + 1,
        callbacks=callbacks,
        max_queue_size=6,
        workers=2)

    del model
    tf.keras.backend.clear_session()
    return experiment_dir, model_dir, args.exp_name


if __name__ == '__main__':
    train()
