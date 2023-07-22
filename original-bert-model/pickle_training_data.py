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

print("Loading squadv2_dev_tf")
# tf_dataset_path = script_path.joinpath(
#     "/work/06333/edbrown/ls6/266/266_final_project_summer_2023/models/squadv2_train_tf"
# )
tf_dataset_path = script_path.joinpath("../cache/squadv2_train_tf")
ds_train = tf.data.Dataset.load(str(tf_dataset_path))
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

print("Prepare data...")
# sample dataset for predictions
total_num_samples = ds_train.cardinality().numpy()
samples = ds_train.take(total_num_samples)
for sample_percent in [20, 40, 60, 80]:
    sample_decimal=sample_percent/100
    batch_num_samples = int(total_num_samples * sample_decimal)
    batch_num_samples_with_answer = int(batch_num_samples * 0.5)
    batch_num_samples_without_answer = int(batch_num_samples * 0.5)
    print(f"batch_num_samples: {batch_num_samples}\nbatch_num_samples_with_answer: {batch_num_samples_with_answer}\nbatch_num_samples_without_answer: {batch_num_samples_without_answer}")


    input_ids = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    impossible = []
    qas_id = []
    start_positions = []
    end_positions = []

    count_num_samples_with_answer = 0
    count_num_samples_without_answer = 0
    for sample in samples:
        if count_num_samples_with_answer < batch_num_samples_with_answer or count_num_samples_without_answer < batch_num_samples_without_answer:
            if sample[1]["is_impossible"].numpy() == 0:
                if count_num_samples_without_answer < batch_num_samples_without_answer:
                        input_ids.append(sample[0]["input_ids"])
                        token_type_ids.append(sample[0]["token_type_ids"])
                        attention_mask.append(sample[0]["attention_mask"])
                        impossible.append(sample[1]["is_impossible"].numpy())
                        qas_id.append(sample[0]["qas_id"].numpy().decode("utf-8"))
                        start_positions.append(sample[1]["start_positions"])
                        end_positions.append(sample[1]["end_positions"])
                count_num_samples_without_answer += 1
            else:
                if count_num_samples_with_answer < batch_num_samples_with_answer:
                    input_ids.append(sample[0]["input_ids"])
                    token_type_ids.append(sample[0]["token_type_ids"])
                    attention_mask.append(sample[0]["attention_mask"])
                    impossible.append(sample[1]["is_impossible"].numpy())
                    qas_id.append(sample[0]["qas_id"].numpy().decode("utf-8"))
                    start_positions.append(sample[1]["start_positions"])
                    end_positions.append(sample[1]["end_positions"])
                count_num_samples_with_answer += 1
        else:
            break

    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
    token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)
    start_positions = tf.convert_to_tensor(start_positions, dtype=tf.int64)
    end_positions = tf.convert_to_tensor(end_positions, dtype=tf.int64)

    training_dict = { "input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask, "start_positions": start_positions, "end_positions": end_positions, "qas_id": qas_id, "impossible": impossible }
    joblib.dump(training_dict, f"training_data_{sample_percent}.pkl")
