import argparse
import logging
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from research_code.params import args
from research_code.gradient_accumulator import GradientAccumulator
from research_code.data_generator import DataGeneratorSingleOutput
from research_code.losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border
from research_code.models import make_model
from research_code.evaluate import m_iou, compute_confusion_matrix   

tf.get_logger().setLevel(logging.INFO)

CLASSES = args.class_names

class TrainLoop:
    # TODO: Add callback support
    # Can try TrainingContext() from tensorflow/python/keras/engine/training_v2.py
    def __init__(self, model, optimizer, loss, log_dir, checkpoint_path, accum_steps=1):
        """
        :param loss: loss object or a list of losses the size of model's outputs
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_logger = tf.summary.create_file_writer(os.path.join(log_dir,'train'))
        self.val_logger = tf.summary.create_file_writer(os.path.join(log_dir,'val'))
        self.accum_steps = accum_steps
        self.grad_accum = GradientAccumulator()
        self.best_val = -np.inf
        self.checkpoint_path = checkpoint_path
        
    @tf.function
    def calculate_loss(self, inputs, targets, training):
        outs = self.model(inputs, training=training)
        outs = tf.nest.flatten(outs)
        targets = tf.nest.flatten(targets)
        # NOTE: Go here for examples of more functionality 
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training_eager.py#L159
        output_losses = []
        for i, loss_fn in enumerate(self.loss):
            out_loss = loss_fn(targets[i], outs[i])
            output_losses.append(out_loss)
        return output_losses
    
    @tf.function
    def calculate_gradients(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_values = self.calculate_loss(inputs, targets, training=True)
        return loss_values, tape.gradient(loss_values, self.model.trainable_variables)
    
    @tf.function          
    def apply_gradients(self):
        # TODO: consider adding gradient clipping
        grads_and_vars = []

        for gradient, variable in zip(self.grad_accum.gradients, self.model.trainable_variables):
            if gradient is not None:
                scaled_gradient = gradient / self.accum_steps
                grads_and_vars.append((scaled_gradient, variable))
            else:
                grads_and_vars.append((gradient, variable))
        
        self.optimizer.apply_gradients(grads_and_vars)
        self.grad_accum.reset()
        
    @tf.function
    def step(self, inputs, targets, perform_update):
        loss_values, grads = self.calculate_gradients(inputs, targets)
        self.grad_accum(grads)
        if perform_update:
            self.apply_gradients()
        return loss_values
        
    # TODO: give this a dict {"train": {"loss":n, "metric":n}, "val": {}}
    # parse it and write logs accordingly
    @tf.function
    def write_logs(self, train_loss, val_metric, step):
        with self.train_logger.as_default():
            tf.summary.scalar(name='loss', data=train_loss, step=step)
        with self.val_logger.as_default():
            tf.summary.scalar(name='mIoU', data=val_metric, step=step)
        
    def train(self, train_dataset, val_dataset, epochs):
        for epoch in range(1, epochs+1):
            # Training
            pbar = tqdm(total=len(train_dataset), desc=f"Train | Epoch {epoch}/{epochs}")
            mean_loss = np.nan
            # NOTE: step_num doesn't update if it's a class attribute, so it's here.
            for step_num, (inputs, targets) in enumerate(train_dataset):
                # NOTE: precalculating perform_update here speeds up self.step ALOT
                # FIXME? If epoch len % accum steps != 0 this will skip all the extra batches in the end
                perform_update = step_num % self.accum_steps
                loss_values = self.step(inputs, targets, perform_update)
                #TODO: use sliding weighted mean for current displayed loss
                total_loss = tf.math.reduce_sum(loss_values)
                mean_loss = np.nanmean(np.vstack([total_loss, mean_loss]))
                pbar.set_postfix_str(f"Total loss: {total_loss:.3f}")
                pbar.update(1)
            pbar.close()
            
            # Validation
            pbar = tqdm(total=len(train_dataset), desc=f"Valid | Epoch {epoch}/{epochs}")
            # FIXME: technically, a 0 here is incorrect,
            # but in 10k+ results one zero doesn't matter that much
            # mean_conf_matrix = np.zeros()
            mean_total = np.full((len(CLASSES), len(CLASSES)), np.nan)
            for inputs, targets in val_dataset:
                outputs = self.model(inputs)
                outputs = tf.nest.flatten(outputs)
                outputs = [o.numpy()[0] for o in outputs]
                outputs = np.dstack(outputs)
                # NOTE/FIXME: The label order might become different here is bg is added
                targets_stacked = tf.nest.flatten(targets)
                targets_stacked = np.dstack(targets_stacked)
                targets_stacked = targets_stacked[0]
                # Calculate validation metric
                conf = compute_confusion_matrix(outputs, targets_stacked, len(CLASSES))
                #TODO: use sliding weighted mean for current displayed metric
                mean_total = np.nanmean(np.dstack([mean_total, conf]), axis=-1)
                pbar.update(1)
            pbar.close()
            
            total_miou, perclass_miou = m_iou(conf, CLASSES)
            mean_val_score = np.mean(total_miou)
            tf.print(f"Total: {total_miou}")
            tf.print(perclass_miou)
            if total_miou > self.best_val:
                tf.print(f"Saving model. Prev best: {self.best_val}. Current best: {total_miou}")
                self.best_val = total_miou
                self.model.save_weights(self.checkpoint_path.format(epoch=epoch, metric=total_miou))
            
            # End epoch
            self.write_logs(mean_loss, mean_total, epoch)
            self.grad_accum.reset()
            train_dataset.on_epoch_end()
            val_dataset.on_epoch_end()

    #TODO: add reset() to reset train loop's state


def main():
    # Params
    train_dir = args.train_dir
    val_dir = args.val_dir
    class_names = args.class_names
    batch_size = args.batch_size
    use_crop = args.use_crop
    crop_height = args.crop_height
    crop_width = args.crop_width
    reshape_height = args.reshape_height
    reshape_width = args.reshape_width
    epochs = args.epochs
    do_aug = args.use_aug
    num_classes = len(args.class_names)
    activation = args.activation
    
    input_channels = 3 + args.stacked_channels
    accum_steps = args.accum_steps
    optimizer = Adam(lr=args.learning_rate)
    loss_list = [make_loss('bce_dice')]
    model = make_model(
        (None, None, input_channels), 
        network=args.network, 
        channels=len(args.class_names), 
        activation=activation
    )
    
    if args.weights is None:
        print('No weights passed, training from scratch')
    else:
        print('Loading weights from {}'.format(args.weights))
        model.load_weights(args.weights, by_name=True)
        
    experiment_dir = os.path.join(args.experiments_dir, args.exp_name)
    model_dir = os.path.join(experiment_dir, args.models_dir)
    log_dir = os.path.join(experiment_dir, args.log_dir)

    if os.path.exists(log_dir) and len(os.listdir(log_dir)) > 0:
        raise ValueError(f"Logs aren't empty. Logdir: {log_dir}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    best_model_file = '{}/{}-{}-{}-{:e}'.format(
        model_dir,
        args.network,
        args.loss_function,
        args.crop_width,
        args.learning_rate
    )
    best_model_file = best_model_file + "-{epoch}-{metric:0.3f}.h5"
    
    # Whether to crop
    crop_size = None
    if use_crop:
        crop_size = (crop_height, crop_width)
        print('Using crops of shape ({}, {})'.format(crop_height, crop_width))
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

    # Set up training
    train_loop = TrainLoop(model, optimizer, loss_list, log_dir, best_model_file, accum_steps)
    train_generator = DataGeneratorSingleOutput(
        train_df,
        classes=class_names,
        img_dir=train_dir,
        batch_size=batch_size,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=do_aug,
        validate_pixels=True,
        activation='sigmoid'
    )
    val_generator = DataGeneratorSingleOutput(
        val_df,
        classes=class_names,
        img_dir=val_dir,
        # FIXME: This MUST be one for now 
        batch_size=1,
        shuffle=True,
        reshape_size=(args.reshape_height, args.reshape_width),
        crop_size=crop_size,
        do_aug=False,
        validate_pixels=True,
        activation='sigmoid'
    )
    
    # Train
    train_loop.train(train_generator, val_generator, epochs)

if __name__ == "__main__":
    main()
