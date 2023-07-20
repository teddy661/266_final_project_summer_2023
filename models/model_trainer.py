import os
import pickle

import data_load
import joblib
from models.model_scorer import generate_scoring_dict
from transformers import BertConfig, TFBertModel, WarmUp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


def train_model(model: tf.keras.Model, epochs=1, batch_size=16):
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
    steps_per_epoch = len(input_ids) // batch_size
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = num_train_steps // 10
    initial_learning_rate = 2e-5
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=0,
        decay_steps=num_train_steps,
    )
    warmup_schedule = WarmUp(
        initial_learning_rate=0,
        decay_schedule_fn=linear_decay,
        warmup_steps=warmup_steps,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=warmup_schedule),
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
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
