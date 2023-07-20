import json
import os
from collections import Counter, defaultdict
import data_load

import numpy as np
from transformers import BertConfig, BertTokenizer, TFBertModel


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


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


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def generate_scoring_dict(model):
    # load pkl file
    print("Loading dev data...")
    (
        input_ids,
        token_type_ids,
        attention_mask,
        _,
        _,
        _,
        qas_id,
    ) = data_load.load_dev()
    print("Execute predictions...")
    new_predictions = model.predict(
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
        probabilities.append(np.amax(new_predictions[0][i]) * np.amax(new_predictions[1][i]))

    print("Choose best answer for split answers...")

    # duplicate_ids = [ x for x,  count in collections.Counter(qas_id).items() if count > 1]
    duplicate_ids = sorted(list_duplicates(qas_id))

    scoring_dict = {}
    maxindex = 0
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

    json_name = f"scoring_dict_{model.name}.json"
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(scoring_dict, f, ensure_ascii=False, indent=4)

    print(f"Wrote {json_name}...")
