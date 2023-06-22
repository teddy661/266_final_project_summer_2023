import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import (
    AutoTokenizer,
    BertConfig,
    TFBertForQuestionAnswering,
    TFBertTokenizer,
)

(ds_train, ds_validation), ds_info = tfds.load(
    "squad/v2.0", split=["train", "validation"], shuffle_files=True, with_info=True
)

bert_config = BertConfig.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad", output_hidden_states=True
)


bert_tokenizer = TFBertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

ds_train_np = tfds.as_numpy(ds_train)
ds_validation_np = tfds.as_numpy(ds_validation)

max_seq_length = 512


input_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int64, name="input_ids"
)
input_masks = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int64, name="input_masks"
)
token_type_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int64, name="input_tokens"
)

bert_inputs = {
    "input_ids": input_ids,
    "attention_mask": input_masks,
    "token_type_ids": token_type_ids,
}

bert_model = TFBertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad", config=bert_config
)

bert_outputs = bert_model(bert_inputs)