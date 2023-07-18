import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel


def create_bert_average_pooler_model(
    MODEL_NAME="bert-large-uncased", max_seq_length=386
):
    """
    Creates a BERT QA model using the HuggingFace transformers library
    and base bert-large-uncased model. T
    """
    bert_config = BertConfig.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    )

    bert_model = TFBertModel.from_pretrained(MODEL_NAME, config=bert_config)
    bert_model.trainable = False
    # bert_model.load_weights("./results/bert-large-uncased/training_checkpoints/ckpt_0004.ckpt")

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
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }

    hidden_states = bert_model(bert_inputs).hidden_states
    concat_hidden_states = tf.concat(hidden_states, axis=-1)
    # embeddings = tf.transpose(hidden_states, perm=[1, 2, 3, 0]
    x1 = tf.reduce_mean(concat_hidden_states, axis=-1, keepdims=True)
    x = tf.keras.layers.Dense(2)(x1)
    start, end = tf.split(x, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)

    softmax_start_logits = tf.keras.layers.Softmax()(start)
    softmax_end_logits = tf.keras.layers.Softmax()(end)

    bert_qa_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[softmax_start_logits, softmax_end_logits],
        name="average_pooler",
    )

    return bert_qa_model
