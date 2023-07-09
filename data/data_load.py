import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForQuestionAnswering


def load_dev():
    return load_data(Path.cwd().joinpath("data/squadv2_dev_tf"))


def load_train():
    return load_data(Path.cwd().joinpath("data/squadv2_train_tf"))


def load_data(data_path: Path):
    ds_train = tf.data.Dataset.load(str(data_path))
    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    samples = ds_train.take(ds_train.cardinality().numpy())
    input_ids = []
    token_type_ids = []
    attention_mask = []
    impossible = []
    start_positions = []
    end_positions = []
    qas_id = []
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

    return (
        input_ids,
        token_type_ids,
        attention_mask,
        impossible,
        start_positions,
        end_positions,
        qas_id,
    )


# def load_dev():
#     tf_dataset_path = Path.cwd().joinpath("data/squadv2_dev_tf")
#     ds_dev = tf.data.Dataset.load(str(tf_dataset_path))
#     ds_dev = ds_dev.cache()
#     ds_dev = ds_dev.prefetch(tf.data.AUTOTUNE)

#     samples = ds_dev.take(ds_dev.cardinality().numpy())
#     input_ids = []
#     token_type_ids = []
#     attention_mask = []
#     impossible = []
#     qas_id = []
#     for sample in samples:
#         input_ids.append(sample[0]["input_ids"])
#         token_type_ids.append(sample[0]["token_type_ids"])
#         attention_mask.append(sample[0]["attention_mask"])
#         impossible.append(sample[1]["is_impossible"].numpy())
#         qas_id.append(sample[0]["qas_id"].numpy().decode("utf-8"))

#     input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
#     token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
#     attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int64)

#     return input_ids, token_type_ids, attention_mask, impossible, qas_id


# def load_train():
#
