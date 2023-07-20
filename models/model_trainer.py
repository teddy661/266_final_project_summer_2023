import os
import pickle

import data_load
import joblib
from models.model_scorer import generate_scoring_dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def train_model(model: tf.keras.Model, optimizer="adam", epochs=1, batch_size=16):
    """
    Compile and train the given BERT model, saving the history and generating scoring dict
    """
    print("loading the training data...")
    (
        input_ids,
        token_type_ids,
        attention_mask,
        impossible,
        start_positions,
        end_positions,
        qas_id,
    ) = data_load.load_train()

    # compile the model

    model.compile(
        optimizer=optimizer,
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        ],
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="start_accuracy"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="end_accuracy"),
        ],
    )

    # model.summary()
    history = model.fit(
        [input_ids, token_type_ids, attention_mask],
        [
            start_positions,
            end_positions,
        ],
        batch_size=batch_size,
        epochs=epochs,
    )

    print("Save history...")
    joblib.dump(
        history.history,
        model.name + "_history.pkl",
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    generate_scoring_dict(model)
