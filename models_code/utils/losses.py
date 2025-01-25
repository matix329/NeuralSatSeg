import tensorflow as tf

class DiceLoss:
    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2. * intersection + 1e-6) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-6)
        return 1 - dice