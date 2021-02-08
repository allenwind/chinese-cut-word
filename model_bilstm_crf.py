import os
import glob
import itertools
import collections
import numpy as np
import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from dataset import CharTokenizer
from dataset import load_ctb6_cws, build_sbme_tags
from layers import MaskBiLSTM
from crf import CRF, ModelWithCRFLoss


def load_dataset(file):
    sentences = load_ctb6_cws(file=file)
    X = ["".join(sentence) for sentence in sentences]
    y = build_sbme_tags(sentences, onehot=False)
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
vocab_size = tokenizer.vocab_size

X_train, y_train = preprocess_dataset(X_train, y_train, maxlen, tokenizer)

X_val, y_val = load_dataset("dev.txt")
X_val, y_val = preprocess_dataset(X_val, y_val, maxlen, tokenizer)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs) # 全局mask
x = Embedding(input_dim=vocab_size, output_dim=hdims)(inputs)
x = MaskBiLSTM(hdims)(x, mask=mask)
x = Dense(hdims)(x)
x = Dense(num_classes)(x)
# CRF需要mask来完成不定长序列的处理，这里是手动传入
# 可以设置Embedding参数mask_zero=True，避免手动传入
crf = CRF(trans_initializer="orthogonal")
outputs = crf(x, mask=mask)

base = Model(inputs=inputs, outputs=outputs)

model = ModelWithCRFLoss(base)
model.summary()
model.compile(optimizer="adam")

batch_size = 32
epochs = 5
file = "weights/weights.bilstm.crf"
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val)
)

X_test, y_test = load_dataset("test.txt")
X_test, y_test = preprocess_dataset(X_test, y_test, maxlen, tokenizer)
model.evaluate(X_test, y_test)
model.save_weights(file)

if __name__ == "__main__":
    import dataset
    import evaluation
    from model_utils import CRFBasedTokenizer

    trans = tf.convert_to_tensor(crf.trans)
    trans = np.array(trans, dtype=np.float32)
    print(trans)
    tokenizer = CRFBasedTokenizer(model, tokenizer, maxlen)
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))
