import os

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import sys, os
import pickle
import joblib

tf.get_logger().setLevel("INFO")

from pathlib import Path

from transformers import BertConfig, BertTokenizer, TFBertForQuestionAnswering

# setup for multi-gpu training
mirrored_strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = Path(r"./training_checkpoints")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch}")

# load pkl file
train_examples = joblib.load("train_examples.pkl", pickle.HIGHEST_PROTOCOL)

# Load dataset from cache
ds_train = tf.data.Dataset.load("squadv2_train_tf", compression="NONE")
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

max_seq_length = 512

bert_tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


def create_bert_qa_model(
    MODEL_NAME="bert-large-uncased-whole-word-masking-finetuned-squad",
):
    with mirrored_strategy.scope():
        bert_config = BertConfig.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
        )

        bert_model = TFBertForQuestionAnswering.from_pretrained(
            MODEL_NAME, config=bert_config
        )

        input_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int64, name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int64, name="input_masks"
        )
        token_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int64, name="token_type_ids"
        )

        bert_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

        bert_output = bert_model(bert_inputs)

        start_logits = bert_output.start_logits
        end_logits = bert_output.end_logits

        softmax_start_logits = tf.keras.layers.Softmax()(start_logits)
        softmax_end_logits = tf.keras.layers.Softmax()(end_logits)

        # Need to do argmax after softmax to get most likely index
        bert_qa_model = tf.keras.Model(
            inputs=[input_ids, token_type_ids, attention_mask],
            outputs=[softmax_start_logits, softmax_end_logits],
        )

        bert_qa_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0
            ),
            loss=[
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            ],
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )
    return bert_qa_model


def combine_bert_subwords(bert_tokenizer, input_ids, predictions):
    all_predictions = []
    new_predictions = []
    for x, prediction in enumerate(predictions[0]):
        tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[x])
        token_list = tokens[
            np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1
        ]
        # new_predictions.append(bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(encodings.input_ids[x][np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1])))
        # new_predictions.append(bert_tokenizer.decode(encodings.input_ids[x][np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1], clean_up_tokenization_spaces=True))
        answer = ""
        ptoken = ""
        for i, token in enumerate(token_list):
            if token.startswith("##"):
                answer += token[2:]
            else:
                if i != 0:
                    answer += " "
                answer += token
            ptoken = token
        all_predictions.append(answer)
    return all_predictions


bert_qa_model = create_bert_qa_model()
# tf.keras.utils.plot_model(bert_qa_model, show_shapes=True)
# bert_qa_model.summary()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_fullpath, save_weights_only=True
    ),
]

print("Before training model...")
# sample dataset for predictions
sample = ds_train.take(100)
input_ids = tf.convert_to_tensor([x[0]["input_ids"] for x in sample], dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(
    [x[0]["token_type_ids"] for x in sample], dtype=tf.int64
)
attention_mask = tf.convert_to_tensor(
    [x[0]["attention_mask"] for x in sample], dtype=tf.int64
)
impossible = tf.convert_to_tensor(
    [x[1]["is_impossible"] for x in sample], dtype=tf.int64
)
new_predictions = bert_qa_model.predict(
    [
        input_ids,
        token_type_ids,
        attention_mask,
    ]
)

new_answers = combine_bert_subwords(bert_tokenizer, input_ids, new_predictions)

for i, q in enumerate(new_answers):
    print(f"Question: {train_examples[i].question_text}")
    print(f"Predicted Answer: {q}")
    print(f"Actual Answer: {train_examples[i].answer_text}")
    print(f"Is Impossible: {impossible[i]}")
    print(80 * "=")

# print("Training model...")

# bert_qa_model.fit(
