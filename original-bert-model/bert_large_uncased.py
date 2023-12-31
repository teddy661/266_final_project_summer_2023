import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel


def create_bert_qa_model(
    MODEL_NAME="bert-large-uncased", optimizer=None, max_seq_length=386
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

    input_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="input_ids"
    )
    token_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="token_type_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int64, name="input_masks"
    )


    bert_inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }

    sequence_embeddings = bert_model(bert_inputs).last_hidden_state
    hidden_states = bert_model(bert_inputs).hidden_states

    logits = tf.keras.layers.Dense(2, name="logits")(sequence_embeddings)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start_logits = tf.squeeze(start_logits, axis=-1)
    end_logits = tf.squeeze(end_logits, axis=-1)

    softmax_start_logits = tf.keras.layers.Softmax()(start_logits)
    softmax_end_logits = tf.keras.layers.Softmax()(end_logits)

    # Need to do argmax after softmax to get most likely index
    bert_qa_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[softmax_start_logits, softmax_end_logits],
    )
    return bert_qa_model
