import tensorflow as tf


def get_average_pooler():
    input_layer = tf.keras.layers.Input(shape=(386, 1024, 25), dtype=tf.float32)
    average_pooler_layer = tf.reduce_mean(input_layer, axis=-1, keepdims=True)
    output_layer = tf.keras.layers.Dense(2)(average_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    model = tf.keras.Model(
        inputs=input_layer, outputs=[start, end], name="average_pooler"
    )
    return model
