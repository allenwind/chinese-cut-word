import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from base import TokenizerBase
from viterbi import viterbi_decode as _viterbi_decode
from viterbi import _log_trans
from viterbi import segment_by_tags

def sparse2onehot(x):
    pass

def onehot2sparse(x):
    pass

class Tokenizer(TokenizerBase):
    """基于深度学习模型训练的分词模型的Tokenizer封装"""

    def __init__(self, model, tokenizer, maxlen, method="greedy", trans=None):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        assert method in ("viterbi", "greedy")
        self.method = method
        if self.method == "viterbi":
            self.decode = self.viterbi_decode
        else:
            self.decode = self.greedy_decode

        if trans is None:
            trans = _log_trans
        self.trans = trans

    def model_predict(self, x):
        return self.model.predict(x)

    def find_word(self, sentence):
        size = len(sentence)
        ids = self.tokenizer.transform([sentence])    
        padded_ids = sequence.pad_sequences(
            ids, 
            maxlen=self.maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )
        scores = self.model_predict(padded_ids)[0][:size]
        tags = self.decode(scores) # 最优或局部最优标签序列
        yield from segment_by_tags(tags, sentence)

    def viterbi_decode(self, scores):
        # SBME标签序列解码最优路径
        return _viterbi_decode(scores, self.trans)

    def greedy_decode(self, scores):
        return np.argmax(scores, axis=1).tolist()

class CRFBasedTokenizer(Tokenizer):

    def __init__(self, model, tokenizer, maxlen):
        super().__init__(model, tokenizer, maxlen)
        self.decode = lambda x: x

    def model_predict(self, x):
        tags = self.model.predict(x)
        return tags

class Evaluator(tf.keras.callbacks.Callback):

    def __init__(self):
        self.best_val_acc = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        pass

class SequenceCrossEntropy(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SparseCrossEntropy, self).__init__(**kwargs)

    def call(self, inputs, mask):
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        y_true, y_pred = inputs
        loss = self.compute_loss(y_true, y_pred, mask)
        self.add_loss(loss)
        return y_pred

    def compute_loss(self, y_true, y_pred, mask):
        # 交叉熵作为loss，但mask掉padding部分
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
        return loss

    def compute_output_shape(self, input_shape):
        return input_shape[0]
