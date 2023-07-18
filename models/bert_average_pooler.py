import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel


def create_bert_average_pooler_model(MODEL_NAME="bert-large-uncased", max_seq_length=386):
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
    bert_model.load_weights("./results/bert-large-uncased/training_checkpoints/ckpt_0004.ckpt")

    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int64, name="input_ids")
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
    embeddings = tf.transpose(hidden_states, perm=[1, 2, 3, 0])
    average_pooler_layer = tf.reduce_mean(embeddings, axis=-1, keepdims=True)
    output_layer = tf.keras.layers.Dense(2, name="logits")(average_pooler_layer)
    start, end = tf.split(output_layer, 2, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)

    # sequence_embeddings = bert_model(bert_inputs).last_hidden_state
    # logits = tf.keras.layers.Dense(2, name="logits")(sequence_embeddings)
    # start_logits, end_logits = tf.split(logits, 2, axis=-1)
    # start_logits = tf.squeeze(start_logits, axis=-1)
    # end_logits = tf.squeeze(end_logits, axis=-1)

    softmax_start_logits = tf.keras.layers.Softmax()(start)
    softmax_end_logits = tf.keras.layers.Softmax()(end)

    bert_qa_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[softmax_start_logits, softmax_end_logits],
        name="average_pooler",
    )

    return bert_qa_model
