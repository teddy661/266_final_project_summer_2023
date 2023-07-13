import os
import pickle
from pathlib import Path

import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from transformers import BertTokenizer
from transformers.data.processors.squad import (
    SquadV2Processor, squad_convert_examples_to_features)


def create_features_cache(samples, cached_data_name):
    """
    Creates a cache for the features to speed up loading and debugging
    """

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

    joblib.dump(
        data,
        cached_data_name,
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()
squad_data_dir = script_path.joinpath("../data/squadv2").resolve()

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
max_seq_length = 386
max_query_length = 64
doc_stride = 128

processor = SquadV2Processor()

# generate all dev cache
dev_examples = processor.get_dev_examples(squad_data_dir)
dev_picklefile_name = "dev_examples.pkl"
print(f"Writing Squad V2 Dev Examples to {dev_picklefile_name}")
joblib.dump(
    dev_examples,
    dev_picklefile_name,
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)

dev_dataset_name = "squadv2_dev_tf"
dev_features = squad_convert_examples_to_features(
    examples=dev_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    threads=8,
    return_dataset="tf",
)
print(f"Writing Training Tensorflow Dataset to {dev_dataset_name}")
dev_features.save(dev_dataset_name)
create_features_cache(dev_features, "dev_cached_feature.pkl")


# generate all train cache
train_examples = processor.get_train_examples(squad_data_dir)
train_picklefile_name = "train_examples.pkl"
print(f"Writing Squad V2 Train Examples to {train_picklefile_name}")
joblib.dump(
    train_examples,
    train_picklefile_name,
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)

training_dataset_name = "squadv2_train_tf"
train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=True,
    threads=8,
    return_dataset="tf",
)
print(f"Writing Training Tensorflow Dataset to {training_dataset_name}")
train_features.save(training_dataset_name)
create_features_cache(train_features, "train_cached_feature.pkl")
