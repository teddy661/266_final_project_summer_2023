import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from transformers import BertConfig, BertTokenizer, TFBertModel
from collections import defaultdict, Counter

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
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch}")

# load pkl file
print("Loading dev_examples.pkl")
train_example_path = script_path.joinpath("dev_examples.pkl")
train_examples = joblib.load(train_example_path, pickle.HIGHEST_PROTOCOL)

# Load dataset from cache
print("Loading squadv2_dev_tf")
tf_dataset_path = script_path.joinpath("squadv2_dev_tf")
ds_train = tf.data.Dataset.load(str(tf_dataset_path))
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

max_seq_length = 386

bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


def create_bert_qa_model(
    MODEL_NAME="bert-large-uncased",
):
    with mirrored_strategy.scope():
        bert_config = BertConfig.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
        )

        bert_model = TFBertModel.from_pretrained(MODEL_NAME, config=bert_config)

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

        sequence_embeddings = bert_model(bert_inputs).last_hidden_state

        logits = tf.keras.layers.Dense(2, name="logits")(sequence_embeddings)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

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


def return_prediction_string(bert_tokenizer, input_ids, predictions):
    pass


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


bert_qa_model = create_bert_qa_model()
# tf.keras.utils.plot_model(bert_qa_model, show_shapes=True)
# bert_qa_model.summary()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_fullpath, save_weights_only=True
    ),
]
bert_qa_model.load_weights("../results/training_checkpoints/ckpt_0006.ckpt")
print("Prepare data...")
# sample dataset for predictions
samples = ds_train.take(ds_train.cardinality().numpy())
input_ids = []
token_type_ids = []
attention_mask = []
impossible = []
qas_id = []
for sample in samples:
    input_ids.append(sample[0]["input_ids"])
    token_type_ids.append(sample[0]["token_type_ids"])
    attention_mask.append(sample[0]["attention_mask"])
    impossible.append(sample[1]["is_impossible"].numpy())
    qas_id.append(sample[0]["qas_id"].numpy().decode("utf-8"))

input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)

print("Execute predictions...")
new_predictions = bert_qa_model.predict(
    [
        input_ids,
        token_type_ids,
        attention_mask,
    ]
)

print("Done with Predictions...")
new_answers = combine_bert_subwords(bert_tokenizer, input_ids, new_predictions)

print("Calculate probabilities for split answers...")
probabilities = []
for i, prediction in enumerate(new_predictions[0]):
    probabilities.append(
        np.amax(new_predictions[0][i]) * np.amax(new_predictions[1][i])
    )

print("Choose best answer for split answers...")


# duplicate_ids = [ x for x,  count in collections.Counter(qas_id).items() if count > 1]
def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


duplicate_ids = sorted(list_duplicates(qas_id))

scoring_dict = {}
for d in duplicate_ids:
    maxp = None
    for i in d[1]:
        if maxp == None or probabilities[i] > maxp:
            maxp = probabilities[i]
            maxindex = i
    scoring_dict[qas_id[maxindex]] = new_answers[maxindex]
for i, q in enumerate(new_answers):
    if qas_id[i] not in scoring_dict:
        scoring_dict[qas_id[i]] = q

# diagnose impossible questions Highly inefficient
# for i, q in enumerate(qas_id):
#    answer = ""
#    question = ""
#    for t in train_examples:
#        if t.qas_id == qas_id[i]:
#            answer = t.answer_text
#            question = t.question_text
#            break
#    if impossible[i] == 1:
#        print(f"Index: {i}")
#        print(f"QAS_ID: {qas_id[i]}")
#        print(f"Question: {question}")
#        print(f"Answer: {answer}")
#        print(f"Prediction: {new_answers[i]}")
#        print(80 * "-")

with open("scoring_dict.json", "w", encoding="utf-8") as f:
    json.dump(scoring_dict, f, ensure_ascii=False, indent=4)
print("Wrote scoring_dict.json")

# print("Training model...")

# bert_qa_model.fit(
