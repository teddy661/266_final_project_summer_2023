import os
import pickle
from pathlib import Path

import data_load
import joblib

from models.model_scorer import generate_scoring_dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")

if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()


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
            tf.keras.metrics.SparseCategoricalAccuracy(name="start_acc"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="end_acc"),
        ],
    )

    # model.summary()
    model_path = script_path.joinpath(model.name)
    checkpoint_fullpath = model_path.joinpath("training_checkpoints/ckpt_{epoch:04d}.ckpt")

    history = model.fit(
        [input_ids, token_type_ids, attention_mask],
        [
            start_positions,
            end_positions,
        ],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_fullpath,
                verbose=1,
                save_weights_only=True,
                save_freq="epoch",
            ),
        ],
    )

    print("Save history...")
    joblib.dump(
        history.history,
        str(model_path.joinpath(f"{model.name}_history.pkl")),
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    generate_scoring_dict(model)
