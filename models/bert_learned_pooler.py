import os

import bert_large_uncased
from data_load import get_checkpoint_path

from layers.learned_pooler import LearnedPooler
from models.model_trainer import train_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def create_learned_pooler(epoch_number):
    print(f"loading the base model from checkpoint {epoch_number}...")
    bert_qa_model = bert_large_uncased.create_bert_qa_model()
    if epoch_number > 0:
        checkpoint_path = get_checkpoint_path(epoch_number)
        bert_qa_model.load_weights(checkpoint_path)

    bert_qa_model.trainable = False
    hidden_states = tf.transpose(bert_qa_model.output[0], perm=[1, 2, 3, 0])

    learned_pooler_layer = LearnedPooler()(hidden_states)

    output_layer = tf.keras.layers.Dense(2, name="logits")(learned_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    start = tf.keras.layers.Softmax()(start)
    end = tf.keras.layers.Softmax()(end)

    model = tf.keras.Model(
        inputs=bert_qa_model.input,
        outputs=[start, end],
        name=f"learned_pooler_epochs_{epoch_number}",
    )

    return model


def train_bert_learned_pooler_model():
    epochs = 1
    batch_size = 16

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        for epoch_count in range(5):
            model = create_learned_pooler(epoch_count)
            train_model(model, epochs=epochs, batch_size=batch_size)


train_bert_learned_pooler_model()
