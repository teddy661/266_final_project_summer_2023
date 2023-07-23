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

from bert_large_uncased import create_bert_qa_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

# setup for multi-gpu training
percent_data = 80
training_data = joblib.load(f"training_data_{percent_data}.pkl")
mirrored_strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = script_path.joinpath(f"training_checkpoints_{percent_data}")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch:04d}.ckpt")

bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


def combine_bert_subwords(bert_tokenizer, input_ids, predictions):
    all_predictions = []
    for x in range(len(predictions[0])):
        answer = ""
        token_list = bert_tokenizer.convert_ids_to_tokens(
            input_ids[x][
                np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1
            ]
        )
        if len(token_list) == 0:
            answer = ""
        elif token_list[0] == "[CLS]":
            answer = ""
        else:
            for i, token in enumerate(token_list):
                if token.startswith("##"):
                    answer += token[2:]
                else:
                    if i != 0:
                        answer += " "
                    answer += token
        all_predictions.append(answer)
    return all_predictions


print("Prepare data...")
input_ids = training_data["input_ids"]
token_type_ids = training_data["token_type_ids"]
attention_mask = training_data["attention_mask"]
start_positions = training_data["start_positions"]
end_positions = training_data["end_positions"]

# Change optimizer based on
# https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
# https://arxiv.org/pdf/1810.04805.pdf
epochs = 1
batch_size = 48
# batch_size = 1
steps_per_epoch = len(input_ids) // batch_size
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

    bert_qa_model = create_bert_qa_model(optimizer=optimizer)
    bert_qa_model.trainable = True
    bert_qa_model.compile(
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

history = bert_qa_model.fit(
    [input_ids, token_type_ids, attention_mask],
    [start_positions, end_positions],
    shuffle=True,
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
    f"bert-model-train-history-{percent_data}.pkl",
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)
