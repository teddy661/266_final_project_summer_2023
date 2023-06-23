import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import (
    AutoTokenizer,
    BertConfig,
    TFBertForQuestionAnswering,
    BertTokenizer,
    TFBertTokenizer,
)

(ds_train, ds_validation), ds_info = tfds.load(
    "squad/v2.0", split=["train", "validation"], shuffle_files=True, with_info=True
)

# Doesn't do sentence pairs
# bert_tokenizer = TFBertTokenizer.from_pretrained(
#    "bert-large-uncased-whole-word-masking-finetuned-squad"
# )

bert_tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

ds_train_input = [x for x in tfds.as_numpy(ds_train)]
# ds_validation_input = [x for x in tfds.as_numpy(ds_validation)]

train_question = [x["question"].decode("utf-8") for x in ds_train_input]
train_context = [x["context"].decode("utf-8") for x in ds_train_input]

max_seq_length = 512

# padding to max sequence in batch
train_encodings = bert_tokenizer(
    train_question,
    train_context,
    truncation=True,
    padding=True,
    max_length=max_seq_length,
    return_tensors="tf",
)


def create_bert_qa_model():
    bert_config = BertConfig.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        output_hidden_states=True,
    )

    bert_model = TFBertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad", config=bert_config
    )

    input_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="input_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="input_masks"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="token_type_ids"
    )

    bert_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    bert_output = bert_model(bert_inputs)

    ##TODO Code Oputput