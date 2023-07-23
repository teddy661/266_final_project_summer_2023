import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from transformers import BertConfig, BertTokenizer, TFBertModel

from bert_large_uncased import create_bert_qa_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

checkpoint_dir = script_path.joinpath("training_checkpoints")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch}")

# load pkl file
# print("Loading dev_examples.pkl")
# train_example_path = script_path.joinpath("../cache/dev_examples.pkl")
# train_examples = joblib.load(train_example_path, "r")

max_seq_length = 386

bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


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
# bert_qa_model.load_weights("../results/bert-large-uncased/training_checkpoints/ckpt_0001.ckpt")
# bert_qa_model.load_weights("../results/bert-large-uncased/training_checkpoints/ckpt_0002.ckpt")
# bert_qa_model.load_weights("../results/bert-large-uncased/training_checkpoints/ckpt_0003.ckpt")
print("Load weights...")
bert_qa_model.load_weights("training_checkpoints_20/ckpt_0001.ckpt")
# bert_qa_model.load_weights("../results/bert-large-uncased/training_checkpoints/ckpt_0005.ckpt")
# bert_qa_model.load_weights("../results/bert-large-uncased/training_checkpoints/ckpt_0006.ckpt")

print("Prepare data...")


dev_data = joblib.load("dev_data.pkl", "r")
input_ids = dev_data["input_ids"]
token_type_ids = dev_data["token_type_ids"]
attention_mask = dev_data["attention_mask"]
start_positions = dev_data["start_positions"]
end_positions = dev_data["end_positions"]
qas_id = dev_data["qas_id"]
impossible = dev_data["impossible"]

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

with open(
    "scoring_dict_bert_large_uncased_20_percent_epoch.json", "w", encoding="utf-8"
) as f:
    json.dump(scoring_dict, f, ensure_ascii=False, indent=4)
print("Wrote scoring_dict.json")
