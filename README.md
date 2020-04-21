# Solution to agrivision challenge

Website:
https://www.agriculture-vision.com/  
Leaderboard:
https://competitions.codalab.org/competitions/23732?secret_key=dba10d3a-a676-4c44-9acf-b45dc92c5fcf#results  

Main entrypoint is `run_experiment.py`.  
The following cli params are available:

* `--epochs` sets the amount of epochs to train
* `--reshape_height` and `--reshape_width` to reshape inputs and outputs prior to feeding to the model
* `--use_crop` to perform random crops
* `--crop_height` and `--crop_width` to set random crop size
* `--learning_rate` to set the learning rate
* `--batch_size` to set training batch size
* `--train_dir` is a path to train images directory
* `--val_dir` is a path to val images directory
* `--test_dir` is a path to test images directory
* `--dataset_df` is a path to dataset dataframe
* `--exclude_bad_labels_df` is a path to a dataframe with examples to remove from the training process
* `--experiments_dir` a path to the directory, where the experiments should be stored
* `--class_names` a list of classes for the model to predict
* `--models_dir` a path to model dir within the current experiment's dir
* `--log_dir` a path to log dir within the current experiment's dir
* `--weights` a path to pretrained weights if needed
* `--freeze_till_layer` the layer up to which the model will be frozen during training
* `--show_summary` whether to print the summary of the model to terminal
* `--network` a string representing a network architecture to use
* `--threshold` threshold to use during evaluation
* `--exp_name` current experiment name
* `--use_aug` whether to use training augmentations
* `--tta` whether to use test-time augmentations
* `--activation` choice of activation function to use, affects data loading process
* `--add_classification_head` whether to add a classification head to the chosen model
* `--cls_head_loss_weight` a weight for the classification head's loss function
* `--channels` a list of channels to use as inputs, channels are calculated by `run_experiment.py` on the first run only.

Some parts of the pipeline can be run separately.  
You can run `train.py` to train the model.  
`predict_masks.py` to generate predictions for validaiton  
`evaluate.py` to calculate mIoU metric for validation  
`predict_masks_submission.py` to generate the submission file

Careful, `predict_masks.py` and `evaluate.py` expect the `--experiments_dir` paramenter to be a path to your *current* experiment and `--weghts` paramenter -- the path to the evaluated model's weights.
