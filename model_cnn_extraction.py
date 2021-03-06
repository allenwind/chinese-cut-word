import os
import glob
import itertools
import collections
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from dataset import CharTokenizer
from dataset import load_ctb6_cws

def segment_split(seq, index=False):
    """把seq = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    切分成[[0, 0], [1, 1, 1], [0, 0], [1], [0], [1]]
    """
    if not seq:
        return []
    def segment(seq):
        buf = [seq[0]]
        for label1, label2 in zip(seq[:-1], seq[1:]):
            if label2 == label1:
                buf.append(label2)
            else:
                yield buf
                buf = [label2]
        if buf:
            yield buf

    def segment_to_region(segments):
        idxs = []
        n = 0
        for span in segments:
            idx = []
            for i in span:
                idx.append(n)
                n += 1
            idxs.append(idx)
        return idxs

    segments = list(segment(seq))
    if index:
        return segment_to_region(segments)
    return segments

def index_ioseq(text, flat=True):
    """['中国', '进出口', '银行', '与', '中国', '银行', '加强', '合作']
    ==>[0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    """
    seq = []
    pid = 1
    for word in text:
        pid = 1 - pid
        ids = [pid] * len(word)
        if flat:
            seq.extend(ids)
        else:
            seq.append(ids)
    return seq

class ResidualGatedConv1D(Layer):

    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(ResidualGatedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True

    def build(self, input_shape):
        self.conv1d = Conv1D(
            filters=self.filters * 2,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="same",
        )
        self.layernorm = LayerNormalization()

        if self.filters != input_shape[-1]:
            self.dense = Dense(self.filters, use_bias=False)

        self.alpha = self.add_weight(
            name="alpha", shape=[1], initializer="zeros"
        )

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            inputs = inputs * mask[:, :, None]

        outputs = self.conv1d(inputs)
        gate = tf.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, "dense"):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs

    def compute_output_shape(self, input_shape):
        shape = self.conv1d.compute_output_shape(input_shape)
        return (shape[0], shape[1], shape[2] // 2)

def pad(x, maxlen):
    x = sequence.pad_sequences(
        x, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    return x

def batch_pad(x):
    maxlen = max([len(i) for i in x])
    return pad(x, maxlen)

def batch_paded_generator(X, y, tokenizer, batch_size, epochs):
    X = tokenizer.transform(X)
    batchs = (len(X) // batch_size + 1) * epochs * batch_size
    X = itertools.cycle(X)
    y = itertools.cycle(y)
    gen = zip(X, y)
    batch_X = []
    batch_y = []
    for _ in range(batchs):
        sample_x, sample_y = next(gen)
        batch_X.append(sample_x)
        batch_y.append(sample_y)
        if len(batch_X) == batch_size:
            yield batch_pad(batch_X), batch_pad(batch_y)[:,:,np.newaxis]
            batch_X = []
            batch_y = []

def load_dataset(file):
    X = load_ctb6_cws(file=file)
    y = [index_ioseq(i) for i in X]
    X = ["".join(i) for i in X]
    return X, y

X_train, y_train = load_dataset("train.txt")
tokenizer = CharTokenizer(mintf=1)
tokenizer.fit(X_train)

maxlen = None
hdims = 128
vocab_size = tokenizer.vocab_size

inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=vocab_size, output_dim=hdims, mask_zero=True)(inputs)
x = LayerNormalization()(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=2)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=4)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=8)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
x = ResidualGatedConv1D(hdims, 3, dilation_rate=1)(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)
model.summary()
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

if __name__ == "__main__":
    batch_size = 32
    epochs = 10
    steps_per_epoch = len(X_train) // batch_size + 1
    gen = batch_paded_generator(X_train, y_train, tokenizer, batch_size, epochs)
    model.fit(
        gen,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )
