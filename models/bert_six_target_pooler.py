import os

import models.bert_large_uncased as bert_large_uncased
from layers.learned_pooler import LearnedPooler
from models.data_load import get_checkpoint_path
from models.model_trainer import train_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def create_six_target_pooler(epoch_count):
    print(f"loading the base model from checkpoint {epoch_count}...")
    bert_qa_model = bert_large_uncased.create_bert_qa_model()
    if epoch_count > 0:
        checkpoint_path = get_checkpoint_path(epoch_count)
        bert_qa_model.load_weights(checkpoint_path)

    bert_qa_model.trainable = False
    all_hidden_states = bert_qa_model.output[0]

    target_layers = tf.stack(
        [all_hidden_states[i] for i in range(19, 25)], axis=-1
    )  # last 6 layers

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
        name=f"six_target_pooler_epochs_{epoch_count:02d}",
    )

    return model


def train_bert_six_target_pooler_model():
    epochs = 1
    batch_size = 48

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        for epoch_count in (0, 1, 2, 3, 4, 20, 40, 60, 80):
            model = create_six_target_pooler(epoch_count)
            epoch_count_for_train = epoch_count if epoch_count > 10 else None
            train_model(
                model, epoch_count=epoch_count_for_train, epochs=epochs, batch_size=batch_size
            )
            del model


train_bert_six_target_pooler_model()
