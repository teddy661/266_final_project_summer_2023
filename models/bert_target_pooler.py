import os

import models.bert_large_uncased as bert_large_uncased
from layers.learned_pooler import LearnedPooler
from models.data_load import get_checkpoint_path
from models.model_trainer import train_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# these are generate from model_learned_pooler_weights.ipynb
epoch_to_target_layers = {
    0: [2, 3, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20],
    1: [0, 2, 3, 5, 6, 7, 9, 10, 17, 18, 23, 24],
    2: [2, 3, 5, 6, 7, 8, 11, 16, 17, 18, 23, 24],
    3: [5, 8, 10, 11, 12, 14, 15, 17, 18, 22, 23, 24],
    4: [1, 8, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24],
}


def create_target_pooler(epoch_count):
    print(f"loading the base model from checkpoint {epoch_count}...")
    bert_qa_model = bert_large_uncased.create_bert_qa_model()
    if epoch_count > 0:
        checkpoint_path = get_checkpoint_path(epoch_count)
        bert_qa_model.load_weights(checkpoint_path)

    bert_qa_model.trainable = False
    all_hidden_states = bert_qa_model.output[0]

    target_index = epoch_to_target_layers[epoch_count]
    target_layers = tf.stack([all_hidden_states[i] for i in target_index], axis=-1)

    learned_pooler_layer = LearnedPooler()(target_layers)
    output_layer = tf.keras.layers.Dense(2, name="logits")(learned_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    start = tf.keras.layers.Softmax()(start)
    end = tf.keras.layers.Softmax()(end)

    model = tf.keras.Model(
        inputs=bert_qa_model.input,
        outputs=[start, end],
        name=f"target_pooler_epochs_{epoch_count:02d}",
    )

    return model


def train_bert_target_pooler_model():
    epochs = 1
    batch_size = 16

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        for epoch_count in (0, 1, 2, 3, 4, 20, 40, 60, 80):
            model = create_target_pooler(epoch_count)
            epoch_count_for_train = epoch_count if epoch_count > 10 else None
            train_model(
                model, epoch_count=epoch_count_for_train, epochs=epochs, batch_size=batch_size
            )
            del model


train_bert_target_pooler_model()
