import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from transformers import BertConfig, TFBertModel, WarmUp
from models.model_trainer import train_model


def create_bert_qa_model(MODEL_NAME="bert-large-uncased", max_seq_length=386):
    """
    Creates a BERT QA model using the HuggingFace transformers library
    and base bert-large-uncased model. T
    """
    bert_config = BertConfig.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    )

    bert_model = TFBertModel.from_pretrained(MODEL_NAME, config=bert_config)

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

    sequence_embeddings = bert_model(bert_inputs).last_hidden_state
    hidden_states = bert_model(bert_inputs).hidden_states

    logits = tf.keras.layers.Dense(2, name="logits")(sequence_embeddings)
    start_logits, end_logits = tf.split(logits, 2, axis=-1)
    start = tf.squeeze(start_logits, axis=-1)
    end = tf.squeeze(end_logits, axis=-1)
    start = tf.keras.layers.Softmax()(start)
    end = tf.keras.layers.Softmax()(end)

    # Need to do argmax after softmax to get most likely index
    bert_qa_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[hidden_states, start, end],
    )
    return bert_qa_model


def train_bert_qa_model():
    epochs = 6
    batch_size = 48

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        bert_qa_model = create_bert_qa_model()
        # bert_qa_model.load_weights("./earlier-training/training_checkpoints/ckpt_0004.ckpt")
        bert_qa_model.trainable = True

        train_model(
            bert_qa_model,
            optimizer=get_optimizer_adamW(epochs, batch_size),
            epochs=epochs,
            batch_size=batch_size,
        )


def get_optimizer_adamW(epochs, batch_size):
    steps_per_epoch = 131911 // batch_size
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = num_train_steps // 10
    initial_learning_rate = 2e-5
    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=0,
        decay_steps=num_train_steps,
    )
    warmup_schedule = WarmUp(
        initial_learning_rate=0,
        decay_schedule_fn=linear_decay,
        warmup_steps=warmup_steps,
    )
    return tf.keras.optimizers.AdamW(learning_rate=warmup_schedule)


# train_bert_qa_model()
