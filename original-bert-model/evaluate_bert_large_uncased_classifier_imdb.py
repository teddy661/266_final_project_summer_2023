import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bert_large_uncased_classifier import create_bert_classifier_model

from transformers import BertConfig, BertTokenizer, TFBertModel, WarmUp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

data_dir = script_path.joinpath("../data")
test_data_path = data_dir.joinpath("imdb_test_data.pkl")

for percent_data in [20, 40, 60, 80, 100]:
    if percent_data == 100:
        epochs = 6
    else:
        epochs = 1

    tf.keras.backend.clear_session()
    if "bert_classifier_model" in globals():
        del bert_classifier_model

    test_data = joblib.load(test_data_path)
    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Always choose the first epoch weights. We know we're overfitting after the first epoch.
    checkpoint_dir = script_path.joinpath(
        f"training_checkpoints_classifier_{percent_data}"
    )
    checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_0001.ckpt")

    test_input_ids = test_data["input_ids"]
    test_token_type_ids = test_data["token_type_ids"]
    test_attention_mask = test_data["attention_mask"]
    test_labels = test_data["labels"]

    with mirrored_strategy.scope():
        batch_size = 30
        steps_per_epoch = len(test_input_ids) // batch_size
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

        optimizer = tf.keras.optimizers.AdamW(learning_rate=warmup_schedule)

        bert_classifier_model = create_bert_classifier_model(optimizer=optimizer)
        bert_classifier_model.trainable = False
        bert_classifier_model.load_weights(checkpoint_fullpath)
        bert_classifier_model.compile(
            optimizer=optimizer,
            loss=[
                tf.keras.losses.BinaryCrossentropy(from_logits=False),
            ],
            metrics=[
                tf.metrics.BinaryAccuracy(),
            ],
        )
        bert_classifier_model.compile(
            optimizer=optimizer,
            loss=[
                tf.keras.losses.BinaryCrossentropy(from_logits=False),
            ],
            metrics=[
                tf.metrics.BinaryAccuracy(),
            ],
        )

        results = bert_classifier_model.evaluate(
            x=[test_input_ids, test_token_type_ids, test_attention_mask],
            y=[test_labels],
            batch_size=batch_size,
            verbose=1,
        )
        results.insert(0, percent_data)
        header = [
            "percent_data",
            "loss",
            "binary_accuracy",
        ]

        with open(script_path.joinpath("results_classifier_test_imdb.csv"), "a") as f:
            if percent_data == 20:
                f.write(",".join(header) + "\n")
            f.write(",".join([str(x) for x in results]) + "\n")
