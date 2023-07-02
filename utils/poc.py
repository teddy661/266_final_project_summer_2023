import os

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
import sys, os
import joblib

tf.get_logger().setLevel("INFO")

from pathlib import Path

from transformers import BertConfig, BertTokenizer, TFBertForQuestionAnswering

from SquadV2 import SquadV2

# setup for multi-gpu training
mirrored_strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = Path(r"./training_checkpoints")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch}")

# load data
script_path = Path(__file__).parent.absolute()
train_features_file = script_path.joinpath("train_features.pkl")
if train_features_file.exists():
    print("Loading train features... This is slow... Please wait...")
    ds_train = joblib.load(train_features_file)
else:
    print("No data found. Please run new_squad.py to generate data.")

print("Done loading data...")

squadv2 = SquadV2()
squadv2.load_data()
# This is the ugly way to get the data out of the tfds object
# 100 Samples
ds_train_input = squadv2.get_train_data(num_samples=100)


train_question = [x["question"].decode("utf-8") for x in ds_train_input]
train_context = [x["context"].decode("utf-8") for x in ds_train_input]

##
## if we wanted to return a tensorflow dataset from convert_examples_to_features
## the following would apply
##
# subset = ds_train.take(100)
# input_ids = [x[0]['input_ids'].numpy() for x in subset]
# attention_mask = [x[0]['attention_mask'].numpy() for x in subset]
# token_type_ids = [x[0]['token_type_ids'].numpy() for x in subset]

max_seq_length = 512

bert_tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

# padding to max sequence in batch
train_encodings = bert_tokenizer(
    train_question,
    train_context,
    truncation="only_second",
    padding="max_length",
    max_length=max_seq_length,
    return_tensors="tf",
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
input_ids = tf.convert_to_tensor([x.input_ids for x in ds_train[0:100]], dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(
    [x.token_type_ids for x in ds_train[0:100]], dtype=tf.int64
)
attention_mask = tf.convert_to_tensor(
    [x.attention_mask for x in ds_train[0:100]], dtype=tf.int64
)
new_predictions = bert_qa_model.predict(
    [
        input_ids,
        token_type_ids,
        attention_mask,
    ]
)
# old_predictions = bert_qa_model.predict(
#    [
#        train_encodings.input_ids,
#        train_encodings.token_type_ids,
#        train_encodings.attention_mask,
#    ]
# )

# old_answers = combine_bert_subwords(bert_tokenizer, train_encodings.input_ids, old_predictions)
new_answers = combine_bert_subwords(bert_tokenizer, input_ids, new_predictions)


# TODO:
# For new predictions we'd need to recover the original dataset since it's not stored in the SquadFeatures object

# for i, q in enumerate(train_question):
#    print(f"Question: {q}")
#    print(f"Old Predicted Answer: {old_answers[i]}")
#    print(f"New Predicted Answer: {new_answers[i]}")
#    if ds_train_input[i]["is_impossible"]:
#        print(
#            f"Plausible Answer: {ds_train_input[i]['plausible_answers']['text'][0].decode('utf-8')}"
#        )
#    else:
#        print(f"Answer: {ds_train_input[i]['answers']['text'][0].decode('utf-8')}")
#    print("---")
#    print(f"Is Impossible: {ds_train_input[i]['is_impossible']}")
#    print(f"Context: {train_context[i]}")
#    print(80 * "=")

# print("Training model...")

# bert_qa_model.fit(
