import argparse
import logging
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam
from research_code.params import args
from research_code.gradient_accumulator import GradientAccumulator
from research_code.datasets_tf2 import DataGenerator_angles
from research_code.losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border, mse_masked, angle_rmse
from research_code.models import make_model
tf.get_logger().setLevel(logging.INFO)


class TrainLoop:
    # TODO: Add callback support
    # Can try TrainingContext() from tensorflow/python/keras/engine/training_v2.py
    def __init__(self, model, optimizer, loss, lr_scheduler, log_dir, accum_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.train_logger = tf.summary.create_file_writer(os.path.join(log_dir,'train'))
        self.val_logger = tf.summary.create_file_writer(os.path.join(log_dir,'val'))
        self.accum_steps = accum_steps
        self.grad_accum = GradientAccumulator()
    
    @tf.function()
    def calculate_loss(self, x, y, training):
        y_ = self.model(x, training=training)
        
        return loss(y_true=y, y_pred=y_)

    @tf.function()
    def calculate_gradients(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.calculate_loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    @tf.function()                
    def apply_gradients(self):
        # TODO: add gradient clipping
        grads_and_vars = []

        for gradient, variable in zip(self.grad_accum.gradients, self.model.trainable_variables):
            if gradient is not None:
                scaled_gradient = gradient / (args["n_device"] * args["gradient_accumulation_steps"])
                grads_and_vars.append((scaled_gradient, variable))
            else:
                grads_and_vars.append((gradient, variable))
        
        self.optimizer.apply_gradients(grads_and_vars)
        self.grad_accum.reset()

    @tf.function()
    def step(self, inputs, target):
        loss_value, grads = self.calculate_gradients(self.model, inputs, target)
        self.grad_accum(grads)
        # NOTE: If epochs % accum steps != 0 this will skip all the extra batches
        if self.optimizer.iterations % self.accum_steps == 0:
            self.apply_gradients()
        return loss_value
        
    # TODO: give this a dict {"train": {"loss":n, "metric":n}, "val": {}}
    # parse it and write logs accordingly
    def write_logs(self, data, step):
        with self.train_logger.as_default():
            tf.summary.scalar(name='loss', data=data, step=step)
        with self.val_logger.as_default():
            tf.summary.scalar(name='loss', data=data, step=step)
        
    #TODO: add reset() to reset train loop's state

    def train(self, train_dataset, val_dataset, epochs):
        for epoch in range(epochs):
            for inputs, target in train_dataset:
                loss_value = self.step(inputs, target, accum_steps)
                tf.print(loss_value)
                
            # End epoch
            # TODO: reset gradient accum
            train_dataset.on_epoch_end()

def main():
    # Params
    man_dir = args.manual_dataset_dir
    class_names = args.class_names
    batch_size = args.batch_size
    use_crop = args.use_crop
    input_height = args.input_height
    input_width = args.input_width
    train_df_path = args.train_df
    val_df_path = args.val_df
    out_height = args.out_height
    out_width = args.out_width
    epochs = args.epochs
    log_dir = os.path.join('./logs', args.exp_name)
    
    # Params not in params.py
    input_channels = 3
    do_aug = True
    accum_steps = 1
    optimizer = Adam()
    lr_scheduler = lambda x: 0.001
    model = make_model((None, None, input_channels))
    
    # Read dataset descriptions
    train_df = pd.read_csv(train_df_path)
    val_df = pd.read_csv(val_df_path)
    
    # Whether to crop
    crop_size = None
    if use_crop:
        crop_size = (input_height, input_width)
        print('Using crops of shape ({}, {})'.format(input_height, input_width))
    else:
        print('Using full size images, --use_crop=True to do crops')
    
    # Set up training
    train_loop = TrainLoop(model, optimizer, loss, lr_scheduler, log_dir)
    train_generator = DataGenerator_angles(
        train_df,
        classes=class_names,
        img_dir=man_dir,
        batch_size=batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        aug=do_aug
    )
    val_generator = DataGenerator_angles(
        val_df,
        classes=args.class_names,
        img_dir=man_dir,
        batch_size=batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        aug=False
    )
    
    # Train
    train_loop.train(train_generator, val_generator, epochs, accum_steps)

if __name__ == "__main__":
    main()