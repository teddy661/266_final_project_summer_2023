import tensorflow as tf


class LearnedPooler(tf.keras.layers.Layer):
    """Implementation of learned pooler reported by Tenney 2019
    Original paper: https://arxiv.org/abs/1905.05950
    """

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            trainable=True,
            initializer="random_normal",
            name="weights",
        )
        self.t = self.add_weight(shape=(1), trainable=True, initializer="ones", name="t")

    def call(self, inputs):
        w = tf.nn.softmax(self.w)
        return tf.reduce_sum(tf.multiply(inputs, w), axis=-1) * self.t
