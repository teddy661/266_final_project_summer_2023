import pickle

import joblib
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertConfig, BertTokenizer
from transformers.data.processors.squad import (
    SquadV2Processor,
    squad_convert_examples_to_features,
)

processor = SquadV2Processor()
train_examples = processor.get_train_examples("../data/squadv2/")
dev_examples = processor.get_dev_examples("../data/squadv2/")

tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

max_seq_length = 512
max_query_length = 64
doc_stride = 128

train_features = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=True,
    threads=8
)
joblib.dump(train_features, "train_features.pkl", pickle.HIGHEST_PROTOCOL)

dev_features = squad_convert_examples_to_features(
    examples=dev_examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=doc_stride,
    max_query_length=max_query_length,
    is_training=False,
    threads=8,
)
joblib.dump(dev_features, "dev_features.pkl", pickle.HIGHEST_PROTOCOL)
