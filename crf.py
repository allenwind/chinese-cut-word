import tensorflow as tf
import tensorflow_addons as tfa

# CRF的简单实现，依赖tensorflow_addons.text中的相关函数

class CRF(tf.keras.layers.Layer):
    """CRF的实现，包括trans矩阵和viterbi解码"""

    def __init__(self, lr_multiplier=1, trans_initializer="glorot_uniform", **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.supports_masking = True
        self.lr_multiplier = lr_multiplier
        self.trans_initializer = tf.keras.initializers.get(trans_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 3
        units = input_shape[-1]
        self.trans = self.add_weight(
            name="trans",
            shape=(units, units),
            initializer=self.trans_initializer
        )

    def call(self, inputs, mask=None):
        # 必须要有相应的mask传入
        # 传入方法：
        # 1.手动传入
        # 2.设置Masking层
        # 3.Embedding层设置mask_zero=True
        assert mask is not None
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        viterbi_tags, _ = tfa.text.crf_decode(inputs, self.trans, lengths)
        # (bs, seq_len), (bs, seq_len, units), (bs,), (units, units)
        return viterbi_tags, inputs, lengths, self.trans

class ModelWithCRFLoss(tf.keras.Model):
    """把CRFloss包装成模型，这种写法容易扩展不同的loss"""

    def __init__(self, base, **kwargs):
        super(ModelWithCRFLoss, self).__init__(**kwargs)
        self.base = base
        self.accuracy_fn = tf.keras.metrics.Accuracy(name="accuracy")

    def call(self, inputs):
        return self.base(inputs)

    def summary(self):
        self.base.summary()

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            viterbi_tags, lengths, crf_loss = self.compute_loss(
                x, y, sample_weight, training=True
            )
        grads = tape.gradient(crf_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        mask = tf.sequence_mask(lengths, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)
        results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
        return results

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        viterbi_tags, lengths, crf_loss = self.compute_loss(
            x, y, sample_weight, training=False
        )
        mask = tf.sequence_mask(lengths, y.shape[1])
        self.accuracy_fn.update_state(y, viterbi_tags, mask)
        results = {"crf_loss": crf_loss, "accuracy": self.accuracy_fn.result()}
        return results

    def predict_step(self, data):
        x, *_ = tf.keras.utils.unpack_x_y_sample_weight(data)
        viterbi_tags, *_ = self(x, training=False)
        return viterbi_tags

    def compute_loss(self, x, y, sample_weight, training):
        viterbi_tags, potentials, lengths, trans = self(x, training=training)
        crf_loss, _ = tfa.text.crf_log_likelihood(potentials, y, lengths, trans)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight
        return viterbi_tags, lengths, tf.reduce_mean(-crf_loss)

    def accuracy(self, y_true, y_pred):
        viterbi_tags, potentials, lengths, trans = y_pred
        mask = tf.sequence_mask(lengths, y_true.shape[1])
        return self.accuracy_fn(y_true, viterbi_tags, mask)
