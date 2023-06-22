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

ds_train_np = tfds.as_numpy(ds_train)
ds_validation_np = tfds.as_numpy(ds_validation)
