from keras.applications.vgg16 import VGG16
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, ConvLSTM2D, Flatten, Conv3D
from tensorflow.keras.layers import Activation, SpatialDropout2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense, Multiply, Add, Lambda, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
import tensorflow.keras.backend as KB
import tensorflow as tf
from tensorflow_addons.layers import CorrelationCost

from research_code.resnet50_fixed import ResNet50, ResNet50_multi
from research_code.params import args
from research_code.sel_models.unets import (create_pyramid_features, conv_relu, prediction_fpn_block, conv_bn_relu,
                                            decoder_block_no_bn)
import numpy as np
from research_code.coord_conv import CoordinateChannel2D

COORD_CONV_SETTING = True


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1), activation='relu', kernel_size=3,
                      coord_conv=args.coord_conv):
    if coord_conv:
        prevlayer = CoordinateChannel2D()(prevlayer)
    conv = Conv2D(filters, (kernel_size, kernel_size), padding="same", kernel_initializer="he_normal", strides=strides,
                  name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    if activation:
        conv = Activation(activation, name=prefix + "_activation")(conv)
    return conv


def conv_block_dilated(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), dilation_rate=2, padding="same", kernel_initializer="he_normal", strides=strides,
                  name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1), coord_conv=args.coord_conv):
    if coord_conv:
        prevlayer = CoordinateChannel2D()(prevlayer)
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides,
                  name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: KB.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(KB.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(KB.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])

    return x


def csse_resnet50_fpn_instance(input_shape, channels=1, activation="sigmoid", class_names=args.class_names):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d")(conv1)
    conv2 = resnet_base.get_layer("activation_9").output
    conv2 = csse_block(conv2, "csse_ngle_net9")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_21").output
    conv3 = csse_block(conv3, "csse_21")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_39").output
    conv4 = csse_block(conv4, "csse_39")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_48").output
    conv5 = csse_block(conv5, "csse_48")
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, "up4")
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    output_dict = {}
    for cls in class_names:
        output_dict[cls] = Conv2D(1, (1, 1), activation="sigmoid", name=cls)(x)
    model = Model(resnet_base.input, output_dict)
    return model


def csse_resnet50_fpn_edge(input_shape, channels=1, activation="sigmoid"):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, "up4")
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    semantic = Conv2D(channels, (1, 1), activation=activation, name="semantic")(x)
    instance = Conv2D(channels, (1, 1), activation=activation, name="instance")(x)
    edge = Conv2D(channels, (1, 1), activation=activation, name="edge")(x)
    model = Model(resnet_base.input, [semantic, instance, edge])
    return model


def sse_block(prevlayer, prefix):
    # conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
    #               name=prefix + "_conv")(prevlayer)
    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])

    return conv


def csse_block(x, prefix):
    """
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    """
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


"""
Unet with Mobile net encoder
Uses caffe preprocessing function
"""


def get_unet_resnet(input_shape):
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # resh_conv = Conv2D()
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


def get_csse_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


def csse_resnet50_fpn(input_shape, channels=1, activation="sigmoid", coord_conv=args.coord_conv):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d")(conv1)
    conv2 = resnet_base.get_layer("activation_9").output
    conv2 = csse_block(conv2, "csse_ngle_net9")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_21").output
    conv3 = csse_block(conv3, "csse_21")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_39").output
    conv4 = csse_block(conv4, "csse_39")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_48").output
    conv5 = csse_block(conv5, "csse_48")
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if coord_conv:
        x = CoordinateChannel2D()(x)
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model


def angle_net_csse_resnet50_fpn(input_shape, channels=1):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d")(conv1)
    conv2 = resnet_base.get_layer("activation_9").output
    conv2 = csse_block(conv2, "csse_9")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_21").output
    conv3 = csse_block(conv3, "csse_21")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_39").output
    conv4 = csse_block(conv4, "csse_39")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_48").output
    conv5 = csse_block(conv5, "csse_48")
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    line_prediction = Conv2D(1, (1, 1), activation='sigmoid', name="mask")(x)
    conv8 = Conv2D(channels, (1, 1), activation=None, name="conv")(x)
    prediction = tf.nn.softmax(conv8, axis=-1, name="prediction_anglenet")
    model = Model(resnet_base.input, [prediction, line_prediction])
    return model


def csse_resnet50_fpn_multi(input_shape, channels=1, activation="sigmoid"):
    resnet_input = tuple([input_shape[0], input_shape[1], 3])
    resnet_base = ResNet50(input_shape=input_shape, include_top=False, weights=None)
    resnet_base_we = ResNet50_multi(input_shape=resnet_input, include_top=False)

    conv_weights, conv_bias = resnet_base_we.layers[1].get_weights()
    # getting new_weights
    new_weights = np.random.normal(size=(7, 7, input_shape[-1], 64), loc=0, scale=0.2)
    new_weights[:, :, :3, :] = conv_weights

    for i in resnet_base_we.layers:
        if i.name == 'conv1':
            resnet_base.get_layer(i.name).set_weights([new_weights, conv_bias])
        try:
            resnet_base.get_layer(i.name).set_weights(resnet_base_we.get_layer(i.name).get_weights())
        except:
            continue
    del resnet_base_we
    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d")(conv1)
    conv2 = resnet_base.get_layer("activation_9").output
    conv2 = csse_block(conv2, "csse_ngle_net9")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_21").output
    conv3 = csse_block(conv3, "csse_21")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_39").output
    conv4 = csse_block(conv4, "csse_39")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_48").output
    conv5 = csse_block(conv5, "csse_48")

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)

    model = Model(resnet_base.input, x)

    return model


def resnet50_fpn(input_shape, channels=1, activation="sigmoid"):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model


def get_csse_hypercolumn_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    hyper = concatenate([conv10,
                         UpSampling2D(size=2)(conv9),
                         UpSampling2D(size=4)(conv8),
                         UpSampling2D(size=8)(conv7),
                         UpSampling2D(size=16)(conv6)], axis=-1)
    hyper = SpatialDropout2D(0.2)(hyper)
    # x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(hyper)
    x = Conv2D(1, (1, 1), name="no_activation_prediction", activation=None)(hyper)
    x = Activation('sigmoid', name="activation_prediction")(x)
    model = Model(resnet_base.input, x)
    return model


def get_simple_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


def get_instance_unet(input_shape, channels=1, activation="sigmoid"):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    output = Conv2D(channels, (1, 1), activation=activation, name="mask")(conv7)
    model = Model(img_input, output)
    return model


def get_instance_unet_correlation(input_shape, channels=1, activation="sigmoid"):
    max_distance = 20
    img_input = Input(input_shape)
    #corr1 = CorrelationCost(pad=max_distance,
    #                        kernel_size=1,
    #                        max_displacement=max_distance,
    #                        stride_1=1,
    #                        stride_2=2,
    #                        data_format="channels_last")([img_input, img_input])
    #corr1 = np.concatenate([img_input, corr1], axis=-1)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)


    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)
    #corr2 = CorrelationCost(pad=max_distance//2,
    #                        kernel_size=1,
    #                        max_displacement=max_distance//2,
    #                        stride_1=1,
    #                        stride_2=2,
    #                        data_format="channels_last")([conv2, conv2])

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)
    corr3 = CorrelationCost(pad=max_distance,
                            kernel_size=1,
                            max_displacement=max_distance,
                            stride_1=1,
                            stride_2=2,
                            data_format="channels_last")([conv3, conv3])

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")
    corr4 = CorrelationCost(pad=max_distance,
                            kernel_size=1,
                            max_displacement=max_distance,
                            stride_1=1,
                            stride_2=2,
                            data_format="channels_last")([conv4, conv4])
    conv4 = concatenate([conv4, corr4], axis=-1)
    up5 = concatenate([UpSampling2D()(conv4), conv3, corr3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    output = Conv2D(channels, (1, 1), activation=activation, name="mask")(conv7)
    model = Model(img_input, output)
    return model


def get_csse_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    conv1 = csse_block(conv1, "csse_1")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    conv2 = csse_block(conv2, "csse_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    conv3 = csse_block(conv3, "csse_3")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")
    conv5 = csse_block(conv5, "csse_5")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


def add_classification_head(input_shape, segmentation_model, encoder_output_name, num_classes):
    model_base = make_model(input_shape, segmentation_model)
    encoder_output = model_base.get_layer(encoder_output_name).output
    x = GlobalAveragePooling2D(name="avg_pool_classification_head")(encoder_output)
    classes = Dense(num_classes, activation='sigmoid', name='classes_prediction')(x)
    final_model = Model(inputs=[model_base.input], outputs=[model_base.output, classes])
    return final_model


def make_model(input_shape, network, **kwargs):
    if network == 'resnet50':
        return get_unet_resnet(input_shape)
    elif network == 'csse_resnet50':
        return get_csse_unet_resnet(input_shape)
    elif network == 'hypercolumn_resnet':
        return get_csse_hypercolumn_resnet(input_shape)
    elif network == 'simple_unet':
        return get_simple_unet(input_shape)
    elif network == 'instance_unet':
        return get_instance_unet(input_shape, **kwargs)
    elif network == 'instance_unet_correlation':
        return get_instance_unet_correlation(input_shape, **kwargs)
    elif network == 'csse_unet':
        return get_csse_unet(input_shape)
    elif network == 'resnet50_fpn':
        return resnet50_fpn(input_shape, **kwargs)
    elif network == 'csse_resnet50_fpn':
        return csse_resnet50_fpn(input_shape, **kwargs)
    elif network == 'csse_resnet50_fpn_multi':
        return csse_resnet50_fpn_multi(input_shape, **kwargs)
    elif network == "csse_resnet_50_fpn_instance":
        return csse_resnet50_fpn_instance(input_shape, **kwargs)
    elif network == "csse_resnet_50_fpn_instance_cls_head":
        return add_classification_head(input_shape, "csse_resnet_50_fpn_instance", **kwargs)
    else:
        raise ValueError("Unknown network")
