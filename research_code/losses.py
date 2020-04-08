import tensorflow.keras.backend as K
# from tensorflow.keras.backend import _to_tensor
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
import tensorflow as tf


def angle_rmse(pred, labels):
    # calculate mask
    pred = tf.cast(tf.argmax(pred, axis=-1), tf.float32)
    labels = tf.cast(tf.argmax(labels, axis=-1), tf.float32)
    mask = tf.cast(tf.not_equal(labels, 0), tf.float32)

    # apply mask
    labels = labels * mask
    pred = pred * mask

    # calculate score
    score = tf.math.sqrt(mean_squared_error(y_pred=pred, y_true=labels))
    score = tf.reduce_mean(score, axis=1)
    return score


def lstm_rmse(pred, labels):
    # calculate mask
    pred = tf.cast(tf.argmax(pred, axis=-1), tf.float32)
    labels = tf.cast(tf.argmax(labels, axis=-1), tf.float32)
    mask = tf.cast(tf.not_equal(labels, 0), tf.float32)

    # apply mask
    labels = labels * mask
    pred = pred * mask

    # calculate score
    score = tf.math.sqrt(mean_squared_error(y_pred=pred, y_true=labels))
    score = tf.reduce_mean(score, axis=1)
    return score


def kld_loss_masked(labels, predictions):
    mask = tf.cast(tf.not_equal(tf.reduce_sum(labels, axis=-1), 0), tf.float64)
    kl_loss = tf.losses.kullback_leibler_divergence(labels, predictions)
    kl_loss = kl_loss * mask
    kl_loss = tf.reduce_sum(kl_loss, axis=0)
    return tf.reduce_mean(kl_loss)


def dice_coef_clipped(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_without_background(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true[..., :-1])
    y_pred_f = K.flatten(y_pred[..., :-1])
    intersection = K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f + y_pred_f)
    return (2. * intersection + smooth) / (denominator + smooth)


def dice_loss_without_background(y_true, y_pred):
    return 1 - dice_without_background(y_true, y_pred)


def bce_dice_softmax(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + dice_loss_without_background(y_true, y_pred)


def bootstrapped_crossentropy(y_true, y_pred, bootstrap_type='hard', alpha=0.95):
    target_tensor = y_true
    prediction_tensor = y_pred
    _epsilon = tf.convert_to_tensor(K.epsilon(), prediction_tensor.dtype.base_dtype)
    prediction_tensor = tf.clip_by_value(prediction_tensor, _epsilon, 1 - _epsilon)
    prediction_tensor = tf.math.log(prediction_tensor / (1 - prediction_tensor))

    if bootstrap_type == 'soft':
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.sigmoid(prediction_tensor)
    else:
        bootstrap_target_tensor = alpha * target_tensor + (1.0 - alpha) * K.cast(
            K.sigmoid(prediction_tensor) > 0.5, tf.float32)
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=bootstrap_target_tensor, logits=prediction_tensor))


def wing_loss(landmarks, labels, w=10.0, epsilon=2.0):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - tf.math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss


def cos_loss_angle(y_true, y_pred):
    """
    (𝜙,𝜃)=2(1−cos(𝜙−𝜃))=(𝜙−𝜃)2+𝑂((𝜙−𝜃)4)
    :return:
    """
    loss = 2 * (1 - tf.math.cos(y_pred - y_true))
    loss = tf.reduce_sum(loss)
    return loss


def kld_loss(labels, predictions):
    """
    Loss for angle net
    :param labels:
    :param predictions:
    :return:
    """
    kl_loss = tf.losses.kullback_leibler_divergence(labels, predictions)
    kl_loss = tf.reduce_sum(kl_loss, axis=0)
    return tf.reduce_mean(kl_loss)


def masked_cos_loss_angle(y_true, y_pred):
    """
    (𝜙,𝜃)=2(1−cos(𝜙−𝜃))=(𝜙−𝜃)2+𝑂((𝜙−𝜃)4)
    :return:
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    # y_true = tf.gather(y_true, tf.where(mask > 0.0, y_true))
    #
    # y_pred = tf.gather_nd(y_pred, tf.where(mask > 0.0, y_pred), )

    y_true = y_true * mask
    y_pred = y_pred * mask
    # 175 degree threshold to change loss in radians
    thresh_175 = 3.05433
    thresh_5 = 0.0872665
    # loss = 2 * (1 - tf.math.cos(y_pred - y_true)) + 4 - 4 * (1 - tf.math.square(tf.math.cos(y_pred - y_true)))
    # losses = tf.where(
    #     tf.logical_or(tf.greater(y_true, thresh_175), tf.greater(thresh_5, y_true)),
    #     4 - 4 * (1 - tf.math.square(tf.math.cos(y_pred - y_true))),
    #     2 * (1 - tf.math.cos(y_pred - y_true))
    # )
    losses = 2 * (1 - tf.math.cos(y_pred - y_true))
    loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
    # loss = tf.reduce_sum(losses)
    return loss


def online_bootstrapping(y_true, y_pred, pixels=512, threshold=0.5):
    """ Implements nline Bootstrapping crossentropy loss, to train only on hard pixels,
        see  https://arxiv.org/abs/1605.06885 Bridging Category-level and Instance-level Semantic Image Segmentation
        The implementation is a bit different as we use binary crossentropy instead of softmax
        SUPPORTS ONLY MINIBATCH WITH 1 ELEMENT!
    # Arguments
        y_true: A tensor with labels.

        y_pred: A tensor with predicted probabilites.

        pixels: number of hard pixels to keep

        threshold: confidence to use, i.e. if threshold is 0.7, y_true=1, prediction=0.65 then we consider that pixel as hard
    # Returns
        Mean loss value
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    difference = K.abs(y_true - y_pred)

    values, indices = K.tf.nn.top_k(difference, sorted=True, k=pixels)
    min_difference = (1 - threshold)
    y_true = K.tf.gather(K.gather(y_true, indices), K.tf.where(values > min_difference))
    y_pred = K.tf.gather(K.gather(y_pred, indices), K.tf.where(values > min_difference))

    return K.mean(K.binary_crossentropy(y_true, y_pred))


def dice_coef_loss_border(y_true, y_pred):
    return (1 - dice_coef_border(y_true, y_pred)) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)


def bce_dice_loss_border(y_true, y_pred):
    return bce_border(y_true, y_pred) * 0.05 + 0.95 * dice_coef_loss(y_true, y_pred)


def dice_coef_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)

    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.gather(y_true_f, tf.where(border > 0.5))
    y_pred_f = K.gather(y_pred_f, tf.where(border > 0.5))

    return dice_coef(y_true_f, y_pred_f)


def bce_border(y_true, y_pred):
    border = get_border_mask((21, 21), y_true)

    border = K.flatten(border)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.gather(y_true_f, tf.where(border > 0.5))
    y_pred_f = K.gather(y_pred_f, tf.where(border > 0.5))

    return binary_crossentropy(y_true_f, y_pred_f)


def get_border_mask(pool_size, y_true):
    negative = 1 - y_true
    positive = y_true
    positive = K.pool2d(positive, pool_size=pool_size, padding="same")
    negative = K.pool2d(negative, pool_size=pool_size, padding="same")
    border = positive * negative
    return border


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def dice_coef_loss_bce_weighted(y_true, y_pred, dice=0.5, bce=0.5, bootstrapping='hard', alpha=1.):
    return bootstrapped_crossentropy(y_true, y_pred, bootstrapping, alpha) * bce + dice_coef_loss(y_true, y_pred) * dice


def mse_masked(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    # y_true = tf.gather(y_true, tf.where(mask > 0.0, y_true))
    #
    # y_pred = tf.gather_nd(y_pred, tf.where(mask > 0.0, y_pred), )

    y_true = y_true * mask
    y_pred = y_pred * mask
    return mean_squared_error(y_true, y_pred)


def wing_masked(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    # y_true = tf.gather(y_true, tf.where(mask > 0.0, y_true))
    #
    # y_pred = tf.gather_nd(y_pred, tf.where(mask > 0.0, y_pred), )

    y_true = y_true * mask
    y_pred = y_pred * mask

    return wing_loss(y_pred, y_true)


def crossentropy_with_KL(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    kl = KL_to_uniform(y_pred)
    return bce + 0.3 * kl


def KL_to_uniform(y_pred):
    channels = 2
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    uniform = K.ones_like(y_pred) / K.cast(channels, K.floatx())
    return uniform * K.log(uniform / y_pred)


def time_crossentropy(labels, pred):
    loss = 0
    for i in range(pred.shape[1]):
        loss += dice_coef_loss_bce(labels[:, i], pred[:, i], dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)
        # loss += K.sum(tf.losses.kullback_leibler_divergence(labels[:, i] > 0.02, pred[:, i]))
        # loss += K.binary_crossentropy(labels[:, i], pred[:, i])
    return loss


def make_loss(loss_name):
    if loss_name == 'crossentropy':
        return K.binary_crossentropy
    elif loss_name == 'crossentropy_time':
        return time_crossentropy
    if loss_name == 'crossentropy_with_kl':
        return crossentropy_with_KL
    elif loss_name == 'crossentropy_boot':
        def loss(y, p):
            return bootstrapped_crossentropy(y, p, 'hard', 0.9)

        return loss
    elif loss_name == 'dice':
        return dice_coef_loss
    elif loss_name == 'bce_dice':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)

        return loss
    elif loss_name == 'bce_dice_softmax':
        return bce_dice_softmax
    elif loss_name == 'boot_soft':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=0.95)

        return loss
    elif loss_name == 'boot_hard':
        def loss(y, p):
            return dice_coef_loss_bce(y, p, dice=0.8, bce=0.2, bootstrapping='hard', alpha=0.95)

        return loss
    elif loss_name == 'online_bootstrapping':
        def loss(y, p):
            return online_bootstrapping(y, p, pixels=512, threshold=0.7)

        return loss
    elif loss_name == 'dice_coef_loss_border':
        return dice_coef_loss_border
    elif loss_name == 'bce_dice_weighted':
        def loss(y, p):
            return dice_coef_loss_bce_weighted(y, p, dice=0.8, bce=0.2, bootstrapping='soft', alpha=1)

        return loss
    elif loss_name == 'bce_dice_loss_border':
        return bce_dice_loss_border
    elif loss_name == 'mean_squared_error':
        return mean_squared_error
    elif loss_name == 'mean_squared_error_masked':
        return mse_masked
    elif loss_name == 'mean_absolute_error':
        return mean_absolute_error
    elif loss_name == 'wing':
        return wing_masked
    elif loss_name == 'cos_loss':
        return cos_loss_angle
    elif loss_name == 'masked_cos_loss':
        return masked_cos_loss_angle
    elif loss_name == 'kl_loss':
        return kld_loss
    elif loss_name == 'kl_loss_masked':
        return kld_loss_masked
    elif loss_name == 'angle_rmse':
        return angle_rmse
    else:
        ValueError("Unknown loss.")
