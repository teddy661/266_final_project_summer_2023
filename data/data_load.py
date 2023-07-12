import os
import pickle
from pathlib import Path

import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForQuestionAnswering


def load_dev():
    return load_data((Path(__file__).parent.joinpath("squadv2_dev_tf")).resolve())


def load_train():
    return load_data((Path(__file__).parent.joinpath("squadv2_train_tf")).resolve())


def load_data(data_path: Path):
    """
    load from cached files directly instead of using tf.data.Dataset.load()
    """

    cached_data_path = data_path.joinpath("cached_feature.pkl").resolve()
    if cached_data_path.exists():
        with open(cached_data_path, "rb") as f:
            return pickle.load(f)

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

    data = (
        input_ids,
        token_type_ids,
        attention_mask,
        impossible,
        start_positions,
        end_positions,
        qas_id,
    )

    if not cached_data_path.exists():
        joblib.dump(
            data,
            cached_data_path,
            compress=False,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return data
