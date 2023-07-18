import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import data_load
import joblib
import numpy as np
import pandas as pd
from bert_average_pooler import create_bert_average_pooler_model
from transformers import BertConfig, BertTokenizer, TFBertModel, WarmUp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")


if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

# setup for multi-gpu training
mirrored_strategy = tf.distribute.MirroredStrategy()
checkpoint_dir = script_path.joinpath("training_checkpoints")
checkpoint_fullpath = checkpoint_dir.joinpath("ckpt_{epoch:04d}.ckpt")

# Change optimizer based on
# https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
# https://arxiv.org/pdf/1810.04805.pdf

(
    train_input_ids,
    train_token_type_ids,
    train_mask,
    train_impossible,
    train_start_positions,
    train_end_positions,
    qas_id,
) = data_load.load_train()

epochs = 6
batch_size = 4

with mirrored_strategy.scope():
    bert_qa_model = create_bert_average_pooler_model()
    # bert_qa_model.trainable = True
    bert_qa_model.compile(
        optimizer="adam",
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        ],
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="start_accuracy"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="end_accuracy"),
        ],
    )
    # bert_qa_model.summary()
    # tf.keras.utils.plot_model(
    #     bert_qa_model, to_file="bert_qa_model.png", show_shapes=True
    # )

    history = bert_qa_model.fit(
        [train_input_ids, train_token_type_ids, train_mask],
        [
            train_start_positions,
            train_end_positions,
        ],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_fullpath,
                verbose=1,
                save_weights_only=True,
                save_freq="epoch",
            ),
        ],
    )

print("Save history...")
joblib.dump(
    history.history,
    "bert-model-train-history.pkl",
    compress=False,
    protocol=pickle.HIGHEST_PROTOCOL,
)
