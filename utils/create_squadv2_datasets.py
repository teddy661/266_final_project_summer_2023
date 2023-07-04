import pickle
from pathlib import Path

import joblib
import tensorflow as tf
from transformers import BertConfig, BertTokenizer
from transformers.data.processors.squad import (
    SquadV2Processor,
    squad_convert_examples_to_features,
)

if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()
squad_data_dir = script_path.joinpath("../data/squadv2/")

processor = SquadV2Processor()
train_examples = processor.get_train_examples(squad_data_dir)
dev_examples = processor.get_dev_examples(squad_data_dir)

train_picklefile_name = "train_examples.pkl"
print(f"Writing Squad V2 Train Examples to {train_picklefile_name}")
joblib.dump(
    train_examples,
    train_picklefile_name,
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)

dev_picklefile_name = "dev_examples.pkl"
print(f"Writing Squad V2 Dev Examples to {dev_picklefile_name}")
joblib.dump(
    dev_examples,
    dev_picklefile_name,
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased"
)

max_seq_length = 512
max_query_length = 64
doc_stride = 128

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
