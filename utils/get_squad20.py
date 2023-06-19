import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFBertForQuestionAnswering, TFBertTokenizer

(ds_train, ds_validation), ds_info = tfds.load(
    "squad/v2.0", split=["train", "validation"], shuffle_files=True, with_info=True
)

bert_model = TFBertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
bert_tokenizer = TFBertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

fetched_train = ds_train.take(ds_train.cardinality().numpy())

bert_train_tokenized = bert_tokenizer(
    fetched_train["context"],
    fetched_train["question"],
    padding=True,
    truncation=True,
    max_length=384,
)
bert_train_inputs = {
    k: bert_train_tokenized[k]
    for k in ["input_ids", "token_type_ids", "attention_mask"]
}
bert_train_labels = {
    "start_positions": bert_train_tokenized["start_positions"],
    "end_positions": bert_train_tokenized["end_positions"],
}
