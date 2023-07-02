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
    threads=8,
    return_dataset="tf",
)
train_features.save("squadv2_train_tf", compression="NONE")
# joblib.dump(train_features, "train_features.pkl", pickle.HIGHEST_PROTOCOL)

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
dev_features.save("squadv2_dev_tf", compression="NONE")
# joblib.dump(dev_features, "dev_features.pkl", pickle.HIGHEST_PROTOCOL)
