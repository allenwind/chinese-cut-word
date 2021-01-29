import tensorflow as tf
from tensorflow.keras.layers import *

class MaskBiLSTM(tf.keras.layers.Layer):
    """支持mask的BiLSTM，比原始BiLSTM开启mask快很多"""

    def __init__(self, hdims, **kwargs):
        super(MaskBiLSTM, self).__init__(**kwargs)
        self.hdims = hdims
        self.forward_lstm = LSTM(hdims, return_sequences=True)
        self.backend_lstm = LSTM(hdims, return_sequences=True)

    def reverse_sequence(self, x, mask):
        seq_len = tf.reduce_sum(mask, axis=1)[:, 0]
        seq_len = tf.cast(seq_len, tf.int32)
        x = tf.reverse_sequence(x, seq_len, seq_axis=1)
        return x

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        else:
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
        x = inputs
        x_forward = self.forward_lstm(x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.backend_lstm(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = x * mask
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.hdims * 2,)
