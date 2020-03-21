import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
# from models import conv_block_simple


def get_angle_net(input_shape, grid_size=8, num_angles=180):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 64, "conv1_1")
    conv1 = conv_block_simple(conv1, 64, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 128, "conv4_1")
    conv4 = conv_block_simple(conv4, 128, "conv4_2")

    x = conv_block_simple(conv4, 256, 'descr_conv1')
    x = conv_block_simple(x, num_angles, 'descr_conv2', kernel_size=1, activation=None)

    #with tf.device('/cpu:0'):  # op not supported on GPU yet
    desc = Lambda(tf.image.resize(x, grid_size * tf.shape(x)[1:3], method='bicubic'))
    prediction = tf.nn.softmax(desc, axis=-1, name="prediction")

    #prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


# def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
#               batch_normalization=True, kernel_reg=0., **params):
#     with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
#         x = tfl.Conv2D(inputs, filters, kernel_size, name='conv',
#                        kernel_regularizer=tf.keras.regularizers.l2(kernel_reg),
#                        data_format=data_format, **params)
#         if batch_normalization:
#             x = tfl.BatchNormalization(
#                     x, training=training, name='bn', fused=True,
#                     axis=1 if data_format == 'channels_first' else -1)
#     return x


#
# def descriptor_head(inputs, **config):
#     params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
#                    'batch_normalization': True,
#                    'training': config['training'],
#                    'kernel_reg': config.get('kernel_reg', 0.)}
#     cfirst = config['data_format'] == 'channels_first'
#     cindex = 1 if cfirst else -1  # index of the channel
#
#     with tf.compat.v1.variable_scope('descriptor', reuse=tf.compat.v1.AUTO_REUSE):
#         x = vgg_block(inputs, 256, 3, 'conv1',
#                       activation=tf.nn.relu, **params_conv)
#         x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
#                       activation=None, **params_conv)
#
#         desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
#         with tf.device('/cpu:0'):  # op not supported on GPU yet
#             desc = tf.image.resize(desc, config['grid_size'] * tf.shape(desc)[1:3], method='bicubic')
#         desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
#         desc = tf.nn.l2_normalize(desc, cindex)
#
#     return {'descriptors_raw': x, 'descriptors': desc}
#
#
#
#
#
# def vgg_backbone(inputs, **config):
#     params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
#                    'activation': tf.nn.relu, 'batch_normalization': True,
#                    'training': config['training'],
#                    'kernel_reg': config.get('kernel_reg', 0.)}
#     params_pool = {'padding': 'SAME', 'data_format': config['data_format']}
#
#     with tf.compat.v1.variable_scope('vgg', reuse=tf.compat.v1.AUTO_REUSE):
#         x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
#         x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
#         x = tfl.MaxPooling2D(x, 2, 2, name='pool1', **params_pool)
#
#         x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
#         x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
#         x = tfl.MaxPooling2D(x, 2, 2, name='pool2', **params_pool)
#
#         x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
#         x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
#         x = tfl.MaxPooling2D(x, 2, 2, name='pool3', **params_pool)
#
#         x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
#         x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)
#
#     return x

#
# class SuperPoint(BaseModel):
#     input_spec = {
#             'image': {'shape': [None, None, None, 1], 'type': tf.float32}
#     }
#     required_config_keys = []
#     default_config = {
#             'data_format': 'channels_first',
#             'grid_size': 8,
#             'detection_threshold': 0.4,
#             'descriptor_size': 180,
#             'batch_size': 32,
#             'learning_rate': 0.001,
#             'lambda_d': 250,
#             'positive_margin': 1,
#             'negative_margin': 0.2,
#             'lambda_loss': 0.0001,
#             'nms': 0,
#             'top_k': 0,
#     }
#
#     def _model(self, inputs, mode, **config):
#
#         def net(image):
#             if config['data_format'] == 'channels_first':
#                 image = tf.transpose(image, [0, 3, 1, 2])
#             features = vgg_backbone(image, **config)
#             descriptors = descriptor_head(features, **config)
#             return descriptors
#
#         results = net(inputs['image'])
#
#         # Apply NMS and get the final prediction
#         prob = results['prob']
#         if config['nms']:
#             prob = tf.map_fn(lambda p: utils.box_nms(
#                 p, config['nms'], keep_top_k=config['top_k']), prob)
#             results['prob_nms'] = prob
#         results['pred'] = tf.to_int32(tf.greater_equal(
#             prob, config['detection_threshold']))
#
#         return results
#
#     def _loss(self, outputs, inputs, **config):
#         logits = outputs['logits']
#         warped_logits = outputs['warped_results']['logits']
#         descriptors = outputs['descriptors_raw']
#         warped_descriptors = outputs['warped_results']['descriptors_raw']
#
#         # Switch to 'channels last' once and for all
#         if config['data_format'] == 'channels_first':
#             logits = tf.transpose(logits, [0, 2, 3, 1])
#             warped_logits = tf.transpose(warped_logits, [0, 2, 3, 1])
#             descriptors = tf.transpose(descriptors, [0, 2, 3, 1])
#             warped_descriptors = tf.transpose(warped_descriptors, [0, 2, 3, 1])
#
#         # Compute the loss for the detector head
#         detector_loss = utils.detector_loss(
#                 inputs['keypoint_map'], logits,
#                 valid_mask=inputs['valid_mask'], **config)
#         warped_detector_loss = utils.detector_loss(
#                 inputs['warped']['keypoint_map'], warped_logits,
#                 valid_mask=inputs['warped']['valid_mask'], **config)
#
#         # Compute the loss for the descriptor head
#         descriptor_loss = utils.descriptor_loss(
#                 descriptors, warped_descriptors, outputs['homography'],
#                 valid_mask=inputs['warped']['valid_mask'], **config)
#
#         tf.summary.scalar('detector_loss1', detector_loss)
#         tf.summary.scalar('detector_loss2', warped_detector_loss)
#         tf.summary.scalar('detector_loss_full', detector_loss + warped_detector_loss)
#         tf.summary.scalar('descriptor_loss', config['lambda_loss'] * descriptor_loss)
#
#         loss = (detector_loss + warped_detector_loss
#                 + config['lambda_loss'] * descriptor_loss)
#         return loss
#
#     def _metrics(self, outputs, inputs, **config):
#         pred = outputs['pred']
#         labels = inputs['keypoint_map']
#
#         precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
#         recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)
#
#         return {'precision': precision, 'recall': recall}