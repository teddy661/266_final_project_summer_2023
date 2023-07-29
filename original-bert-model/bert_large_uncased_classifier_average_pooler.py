import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel


def create_bert_classifier_average_pooler(
    MODEL_NAME="bert-large-uncased",
    optimizer=None,
    max_seq_length=512,
    weights_file=None,
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

    pooler_output = (bert_model(bert_inputs)).pooler_output
    hidden_states = bert_model(bert_inputs).hidden_states
    concat_hidden_states = tf.concat(bert_model(bert_inputs).hidden_states, axis=-1)
    average_pooler_layer = tf.reduce_mean(concat_hidden_states, axis=-1, keepdims=False)
    concat_pooler_output = tf.concat([pooler_output, average_pooler_layer], axis=-1)
    dropout_layer = tf.keras.layers.Dropout(0.1)(concat_pooler_output)
    classifier_layer = tf.keras.layers.Dense(
        1, activation="sigmoid", name="classifier"
    )(dropout_layer)

    bert_classifier_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[classifier_layer],
    )

    if weights_file:
        bert_model.load_weights(weights_file)

    bert_model.trainable = False

    return bert_classifier_model
