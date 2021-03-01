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
from dataset import load_ctb6_cws, build_sbme_tags
from layers import MaskBiLSTM

# 逐时间步预测SMBE标签

def softmax(x, axis=-1):
    # numpy实现的softmax，可以避免溢出
    x = np.array(x)
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)

class SequenceCrossEntropy(tf.keras.losses.Loss):
    """序列上的交叉熵"""

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mask = self.compute_mask(y_true)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def compute_mask(self, y_true):
        mask = tf.not_equal(y_true, 0.0)
        mask = tf.reduce_any(mask, axis=2, keepdims=False)
        mask = tf.cast(mask, tf.float32)
        return mask

class SequenceAccuracy(tf.keras.metrics.Metric):
    """序列上的Accuracy"""

    def __init__(self):
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

def load_dataset(file):
    sentences = load_ctb6_cws(file=file)
    X = ["".join(sentence) for sentence in sentences]
    y = build_sbme_tags(sentences, onehot=True)
    return X, y

def preprocess_dataset(X, y, maxlen, tokenizer):
    X = tokenizer.transform(X)
    X = sequence.pad_sequences(
        X, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
    )
    y = sequence.pad_sequences(
        y, 
        maxlen=maxlen,
        dtype="float32",
        padding="post",
        truncating="post",
        value=0
    )
    return X, y

X_train, y_train = load_dataset("train.txt")
tokenizer = CharTokenizer(mintf=5)
tokenizer.fit(X_train)

maxlen = 128
hdims = 128
num_classes = 4
batch_size = 32
epochs = 5
vocab_size = tokenizer.vocab_size

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer)

X_val, y_val = load_dataset("dev.txt")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
x = LayerNormalization()(x)
x = MaskBiLSTM(hdims)(x, mask=mask)
x = Dense(hdims)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(
    loss=SequenceCrossEntropy(),
    optimizer="adam",
    metrics=["accuracy"]
)

file = "weights/softmax.weights"
if glob.glob(file+".*"):
    model.load_weights(file)
else:
    callbacks = []
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_val, y_val)
    )
    model.save_weights(file)

if __name__ == "__main__":
    import dataset
    import evaluation
    from model_utils import Tokenizer
    tokenizer = Tokenizer(model, tokenizer, maxlen, "viterbi")
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

    # 测试分词的完整性
    text = dataset.load_human_history()[10000:10000+10000]
    words = tokenizer.cut(text)
    assert "".join(words) == text

    evaluation.evaluate_speed(tokenizer.cut, text, rounds=1)
