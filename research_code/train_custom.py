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
from research_code.datasets_tf2 import DataGenerator_angles
from research_code.losses import make_loss, dice_coef_clipped, dice_coef, dice_coef_border, mse_masked, angle_rmse
from research_code.models import make_model
from research_code.evaluate import m_iou, compute_confusion_matrix   

tf.get_logger().setLevel(logging.INFO)

CLASSES = args.class_names

class TrainLoop:
    # TODO: Add callback support
    # Can try TrainingContext() from tensorflow/python/keras/engine/training_v2.py
    def __init__(self, model, optimizer, loss, log_dir, accum_steps=1):
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
    def write_logs(self, data, step):
        with self.train_logger.as_default():
            tf.summary.scalar(name='loss', data=data, step=step)
        with self.val_logger.as_default():
            tf.summary.scalar(name='loss', data=data, step=step)
        
    def train(self, train_dataset, val_dataset, epochs):
        for epoch in range(1, epochs+1):
            # Training
            pbar = tqdm(total=len(train_dataset), desc=f"Train | Epoch {epoch}/{epochs}")
            # NOTE: step_num doesn't update if it's a class attribute, so it's here.
            for step_num, (inputs, targets) in enumerate(train_dataset):
                # NOTE: precalculating perform_update here speeds up self.step ALOT
                # FIXME? If epoch len % accum steps != 0 this will skip all the extra batches in the end
                perform_update = step_num % self.accum_steps
                loss_values = self.step(inputs, targets, perform_update)
                loss_values_dict = dict(zip(CLASSES, loss_values))
                total_loss = tf.math.reduce_sum(loss_values)
                loss_values_dict['Loss'] = total_loss
                pbar.set_postfix({k: f"{v:.3f}" for k, v in loss_values_dict.items()})
                pbar.update(1)
                
            pbar.close()
            
            # TODO: Add validation, model saving, 
            
            # # Validation
            # pbar = tqdm(total=len(train_dataset), desc=f"Valid | Epoch {epoch}/{epochs}")
            # # FIXME: technically, a 0 here is incorrect,
            # # but in 10k+ results one zero doesn't matter that much
            # mean_conf_matrix = np.fill(len())
            # for inputs, targets in val_dataset:
            #     outputs = model(inputs)
            #     outputs = tf.nest.flatten(outputs)
            #     outputs = [o.numpy()[0] for o in outputs]
            #     outputs = np.dstack(outputs, axis=-1)
            #     outputs = outputs > 
            #     background = np.logical_or.recude(, axis=-1)
                
            #     # Calculate validation metric
            #     conf = compute_confusion_matrix(,)
            #     total_ioum, class_ioum = m_iou(conf, len(CLASSES)) 
            #     pbar.set_postfix({k: f"{v:.3f}" for k, v in loss_values_dict.items()})
            #     pbar.update(1)

            
            # End epoch
            self.grad_accum.reset()
            train_dataset.on_epoch_end()
            val_dataset.on_epoch_end()

    #TODO: add reset() to reset train loop's state


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
    num_classes = len(args.class_names)
    log_dir = os.path.join('./logs', args.exp_name)
    
    # Params not in params.py
    input_channels = 3
    do_aug = True
    accum_steps = 2
    optimizer = Adam()
    loss_list = [make_loss('bce_dice') for i in range(num_classes)]
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
    train_loop = TrainLoop(model, optimizer, loss_list, log_dir, accum_steps)
    train_generator = DataGenerator_angles(
        train_df,
        classes=class_names,
        img_dir=man_dir,
        batch_size=batch_size,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        aug=do_aug,
    )
    val_generator = DataGenerator_angles(
        val_df,
        classes=class_names,
        img_dir=man_dir,
        # FIXME: This MUST be one for now 
        batch_size=1,
        shuffle=True,
        out_size=(args.out_height, args.out_width),
        crop_size=crop_size,
        aug=False
    )
    
    # Train
    train_loop.train(train_generator, val_generator, epochs)

if __name__ == "__main__":
    main()
