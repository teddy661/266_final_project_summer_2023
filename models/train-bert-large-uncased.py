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
mirrored_strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = script_path.joinpath("training_checkpoints")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch:04d}.ckpt")

# load pkl file
# print("Loading dev_examples.pkl")
# dev_example_path = script_path.joinpath("dev_examples.pkl")
# dev_examples = joblib.load(dev_example_path, pickle.HIGHEST_PROTOCOL)

# Load dataset from cache
print("Loading squadv2_dev_tf")
tf_dataset_path = script_path.joinpath(
    "/work/06333/edbrown/ls6/266/266_final_project_summer_2023/models/squadv2_train_tf"
)
#tf_dataset_path = script_path.joinpath("../cache/squadv2_train_tf")
ds_train = tf.data.Dataset.load(str(tf_dataset_path))
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


def return_prediction_string(bert_tokenizer, input_ids, predictions):
    pass


def combine_bert_subwords(bert_tokenizer, input_ids, predictions):
    all_predictions = []
    for x in range(len(predictions[0])):
        answer = ""
        token_list = bert_tokenizer.convert_ids_to_tokens(
            input_ids[x][np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1]
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
# sample dataset for predictions
samples = ds_train.take(ds_train.cardinality().numpy())
# samples = ds_train.take(1000)
input_ids = []
input_ids = []
token_type_ids = []
attention_mask = []
impossible = []
qas_id = []
start_positions = []
end_positions = []

for sample in samples:
    input_ids.append(sample[0]["input_ids"])
    token_type_ids.append(sample[0]["token_type_ids"])
    attention_mask.append(sample[0]["attention_mask"])
    impossible.append(sample[1]["is_impossible"].numpy())
    qas_id.append(sample[0]["qas_id"].numpy().decode("utf-8"))
    start_positions.append(sample[1]["start_positions"])
    end_positions.append(sample[1]["end_positions"])

input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
start_positions = tf.convert_to_tensor(start_positions, dtype=tf.int64)
end_positions = tf.convert_to_tensor(end_positions, dtype=tf.int64)

# Change optimizer based on
# https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
# https://arxiv.org/pdf/1810.04805.pdf
epochs = 6
batch_size = 48
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
    bert_qa_model.load_weights("./earlier-training/training_checkpoints/ckpt_0004.ckpt")
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

print('Save history...')
joblib.dump(
    history.history,
    "bert-model-train-history.pkl",
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)
