import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from transformers import BertConfig, BertTokenizer, TFBertModel, WarmUp

from bert_large_uncased_classifier_average_pooler import (
    create_bert_classifier_average_pooler,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

data_dir = script_path.joinpath("../data")
validation_data_path = data_dir.joinpath("imdb_validation_data.pkl")
# setup for multi-gpu training

for percent_data in [20, 40, 60, 80, 100]:
    if percent_data == 100:
        epochs = 6
    else:
        epochs = 1

    tf.keras.backend.clear_session()
    if "bert_classifier_average_pooler_model" in globals():
        del bert_classifier_average_pooler_model
    training_data_path = data_dir.joinpath(f"imdb_training_data_{percent_data}.pkl")
    training_data = joblib.load(training_data_path)
    validation_data = joblib.load(validation_data_path)
    mirrored_strategy = tf.distribute.MirroredStrategy()

    checkpoint_dir = script_path.joinpath(
        f"training_checkpoints_classifier_{percent_data}"
    )
    if percent_data == 100:
        checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_0004.ckpt")
    else:
        checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_0001.ckpt")

    save_checkpoint_dir = script_path.joinpath(
        f"training_checkpoints_classifier_average_pooler{percent_data}"
    )
    save_checkpoint_fullpath = save_checkpoint_dir.joinpath("ckpt_{epoch:04d}.ckpt")

    train_input_ids = training_data["input_ids"]
    train_token_type_ids = training_data["token_type_ids"]
    train_attention_mask = training_data["attention_mask"]
    train_labels = training_data["labels"]

    val_input_ids = validation_data["input_ids"]
    val_token_type_ids = validation_data["token_type_ids"]
    val_attention_mask = validation_data["attention_mask"]
    val_labels = validation_data["labels"]

    batch_size = 30
    steps_per_epoch = len(train_input_ids) // batch_size
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = num_train_steps // 10
    initial_learning_rate = 2e-5

    with mirrored_strategy.scope():
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

        optimizer = tf.keras.optimizers.AdamW(learning_rate=warmup_schedule)

        bert_classifier_average_pooler_model = create_bert_classifier_average_pooler(weights_file=checkpoint_fullpath)
        bert_classifier_average_pooler_model.load_weights(checkpoint_fullpath)
        bert_classifier_average_pooler_model.compile(
            optimizer=optimizer,
            loss=[
                tf.keras.losses.BinaryCrossentropy(from_logits=False),
            ],
            metrics=[
                tf.metrics.BinaryAccuracy(),
            ],
        )

        print(bert_classifier_average_pooler_model.summary())

        # exit()
        history = bert_classifier_average_pooler_model.fit(
            [train_input_ids, train_token_type_ids, train_attention_mask],
            [train_labels],
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [val_input_ids, val_token_type_ids, val_attention_mask],
                [val_labels],
            ),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=save_checkpoint_fullpath,
                    verbose=1,
                    save_weights_only=True,
                    save_freq="epoch",
                ),
            ],
        )

        print("Save history...")
        joblib.dump(
            history.history,
            f"bert-large-uncased-classifier-average-pooler-model-train-history-{percent_data}.pkl",
            compress=False,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
