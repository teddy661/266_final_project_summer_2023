import os

import bert_large_uncased
from data_load import get_checkpoint_path

from models.model_trainer import train_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def create_bert_average_pooler(epoch_count):
    print(f"loading the base model from checkpoint {epoch_count}...")
    bert_qa_model = bert_large_uncased.create_bert_qa_model()
    if epoch_count > 0:
        checkpoint_path = get_checkpoint_path(epoch_count)
        bert_qa_model.load_weights(checkpoint_path)

    bert_qa_model.trainable = False
    hidden_states = tf.transpose(bert_qa_model.output[0], perm=[1, 2, 3, 0])

    average_pooler_layer = tf.reduce_mean(hidden_states, axis=-1)

    output_layer = tf.keras.layers.Dense(2, name="logits")(average_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    start = tf.keras.layers.Softmax()(start)
    end = tf.keras.layers.Softmax()(end)

    model = tf.keras.Model(
        inputs=bert_qa_model.input,
        outputs=[start, end],
        name=f"average_pooler_epochs_{epoch_count:02d}",
    )

    return model


def train_bert_average_pooler_model():
    epochs = 1
    batch_size = 16

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        for epoch_count in [40, 60, 80]:
            model = create_bert_average_pooler(epoch_count)
            train_model(model, epoch_count=epoch_count, epochs=epochs, batch_size=batch_size)
            del model


train_bert_average_pooler_model()
