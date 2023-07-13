import tensorflow as tf


class LearnedPooler(tf.keras.layers.Layer):
    """Implementation of learned pooler reported by Tenney 2019
    Original paper: https://arxiv.org/abs/1905.05950
    """

    def __init__(self, units=1):
        super().__init__()

        # Will only work currently with units = 1
        self.units = 1

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            trainable=True,
            initializer="random_normal",
            name="weights",
        )
        self.t = self.add_weight(
            shape=(1), trainable=True, initializer="ones", name="t"
        )

    def call(self, inputs):
        w = tf.nn.softmax(self.w)
        return tf.reduce_sum(tf.multiply(inputs, w), axis=-1, keepdims=True) * self.t


def get_learned_pooling_model():
    input_layer = tf.keras.layers.Input(shape=(386, 1024, 25), dtype=tf.float32)
    learned_pooler_layer = LearnedPooler()(input_layer)
    output_layer = tf.keras.layers.Dense(2)(learned_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    model = tf.keras.Model(
        inputs=input_layer, outputs=[start, end], name="learned_pooler"
    )
    return model
