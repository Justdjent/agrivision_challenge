import os

import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from research_code.data_generator import DataGeneratorSingleOutput, DataGeneratorClassificationHead
from research_code.losses import make_loss, dice_coef, dice_without_background
from research_code.models import make_model
from research_code.params import args
from research_code.utils import freeze_model
import horovod.tensorflow.keras as hvd


def setup_env():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def is_primary():
    return hvd.rank() == 0


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
        raise ValueError(
            "Please check if this experiment was already run (logs aren't empty) {}".format(experiment_dir))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_model_file = \
        '{}/{}{}loss-{}-{}{:.6f}'.format(model_dir, args.network,
                                         formatted_net_alias, args.loss_function, args.crop_width,
                                         args.learning_rate) + \
        '-{epoch:d}-{val_loss:0.7f}.h5'

    activation = args.activation
    model = make_model((None, None, len(args.channels)),
                       network=args.network,
                       channels=len(args.class_names),
                       activation=activation,
                       add_classification_head=args.add_classification_head,
                       classes=args.class_names)

    freeze_model(model, args.freeze_till_layer)
    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)

    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = Adam(args.learning_rate * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer)

    if args.show_summary:
        model.summary()
    if activation == 'softmax':
        loss_list = [make_loss('bce_dice_softmax')]
        metrics_list = [dice_without_background]
    elif activation == 'sigmoid':
        loss_list = [make_loss('bce_dice')]
        metrics_list = [dice_coef]
    else:
        raise ValueError(f"Unknown activation function - {activation}")

    loss_weights = None
    if args.add_classification_head:
        # if metrics are passed as lists then each metric is calculated for each task
        losses = {}
        metrics = {}
        loss_weights = {}
        # get names of output layers in order to create dict with metrics/losses
        output_names = [layer.name.split('/')[0] for layer in model.output]
        for name in output_names:
            if name == "classification":
                losses[name] = make_loss('crossentropy')
                metrics[name] = [tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                                 tf.keras.metrics.AUC(num_thresholds=20, curve='ROC', name="roc"),
                                 tf.keras.metrics.AUC(num_thresholds=20, curve='PR', name="pr")]
                loss_weights[name] = args.cls_head_loss_weight
            else:
                losses[name] = loss_list[0]
                metrics[name] = metrics_list[0]
                loss_weights[name] = 1 - args.cls_head_loss_weight
        loss_list = losses
        metrics_list = metrics
    if args.add_classification_head:
        generator_class = DataGeneratorClassificationHead
    else:
        generator_class = DataGeneratorSingleOutput

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(loss=loss_list,
                  optimizer=optimizer,
                  metrics=metrics_list,
                  loss_weights=loss_weights,
                  experimental_run_tf_function=False)

    crop_size = None

    if args.use_crop:
        crop_size = (args.crop_height, args.crop_width)
        print('Using crops of shape ({}, {})'.format(args.crop_height, args.crop_width))
    else:
        print('Using full size images, --use_crop=True to do crops')
    dataset_df = pd.read_csv(args.dataset_df)
    print("Total df size {}".format(len(dataset_df)))
    if args.exclude_bad_labels_df:
        invalid_df = pd.read_csv(args.exclude_bad_labels_df)
        print("Number of invalid records {}".format(invalid_df['invalid'].sum()))
        dataset_df = pd.merge(dataset_df, invalid_df, on='name', how='outer')
        dataset_df['invalid'] = dataset_df['invalid'].fillna(False)
        dataset_df = dataset_df[~dataset_df['invalid']]
    print("Total df size {} after cleaning".format(len(dataset_df)))
    train_df = dataset_df[dataset_df["ds_part"] == "train"]

    val_df = dataset_df[dataset_df["ds_part"] == "val"]
    print('{} in train_ids, {} in val_ids, total {}'.format(len(train_df), len(val_df), len(train_df) + len(val_df)))

    train_df = train_df[:100]
    val_df = val_df[:100]
    train_records = len(train_df) // hvd.size()
    val_records = len(val_df) // hvd.size()
    train_df = train_df[hvd.rank() * train_records:(hvd.rank() + 1) * train_records]
    val_df = val_df[hvd.rank() * val_records:(hvd.rank() + 1) * val_records]

    train_generator = generator_class(
        train_df,
        classes=args.class_names,
        img_dir=train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=args.use_aug,
        validate_pixels=True,
        activation=activation,
        channels=args.channels,
        random_state=hvd.rank()
    )

    val_generator = generator_class(
        val_df,
        classes=args.class_names,
        img_dir=val_dir,
        batch_size=args.batch_size // 4,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=False,
        validate_pixels=True,
        activation=activation,
        channels=args.channels,
        random_state=hvd.rank()
    )

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),

        EarlyStopping(patience=25, verbose=10),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(best_model_file, monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         mode='min'))
        TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                    write_images=True)

    verbose = 1 if is_primary() else 0

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_df) // args.batch_size + 1,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_df) // args.batch_size + 1,
        callbacks=callbacks,
        max_queue_size=8,
        workers=4,
        verbose=verbose)

    del model
    tf.keras.backend.clear_session()
    return experiment_dir, model_dir, args.exp_name


if __name__ == '__main__':
    train()
