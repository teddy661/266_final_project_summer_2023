import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel

from layers.learned_pooler import LearnedPooler


def create_bert_classifier_six_target_pooler(
    MODEL_NAME="bert-large-uncased",
    optimizer=None,
    max_seq_length=512,
    train_bert=False,
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
    if train_bert:
        bert_model.trainable = True
    else:
        bert_model.trainable = False

    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int64, name="input_ids")
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
    hidden_states = bert_model(bert_inputs).hidden_states  # 25 * None * 512 * 1024

    dropout_layer = tf.keras.layers.Dropout(0.1)(pooler_output)  # 1024
    classifier_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(
        dropout_layer
    )

    target_layers = tf.stack([hidden_states[i] for i in range(19, 25)], axis=-1)  # last 6 layers

    learned_pooler_layer = LearnedPooler()(target_layers)
    average_pooler_layer = tf.reduce_mean(learned_pooler_layer, axis=1, keepdims=False)
    concat_pooler_output = tf.concat([pooler_output, average_pooler_layer], axis=-1)
    # hidden_states_layer = tf.transpose(hidden_states, perm=[1, 2, 3, 0])
    # average_pooler_layer = tf.reduce_mean(hidden_states_layer, axis=-1)
    new_dropout_layer = tf.keras.layers.Dropout(0.1)(concat_pooler_output)
    pooler_classifier_layer = tf.keras.layers.Dense(
        1, activation="sigmoid", name="pooler_classifier_layer"
    )(new_dropout_layer)

    bert_classifier_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[classifier_layer, pooler_classifier_layer],
    )

    if weights_file:
        print(f"Loading weights from {weights_file}")
        bert_classifier_model.load_weights(weights_file)

    return bert_classifier_model
