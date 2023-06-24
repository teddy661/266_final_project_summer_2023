import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import (
    BertConfig,
    TFBertForQuestionAnswering,
    BertTokenizer,
)

(ds_train, ds_validation), ds_info = tfds.load(
    "squad/v2.0", split=["train", "validation"], shuffle_files=True, with_info=True
)


bert_tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

ds_train_input = [x for x in tfds.as_numpy(ds_train)]
ds_validation_input = [x for x in tfds.as_numpy(ds_validation)]

train_question = [x["question"].decode("utf-8") for x in ds_train_input[:10]]
train_context = [x["context"].decode("utf-8") for x in ds_train_input[:10]]

max_seq_length = 512

# padding to max sequence in batch
train_encodings = bert_tokenizer(
    train_question,
    train_context,
    truncation=True,
    padding="max_length",
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
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }

    bert_output = bert_model(bert_inputs)

    start_logits = bert_output.start_logits
    end_logits = bert_output.end_logits

    softmax_start_logits = tf.keras.layers.Softmax()(start_logits)
    softmax_end_logits = tf.keras.layers.Softmax()(end_logits)

    # Need to do argmax after softmax to get most likely index
    bert_qa_model = tf.keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[softmax_start_logits, softmax_end_logits],
    )

    bert_qa_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return bert_qa_model


def combine_bert_subwords(bert_tokenizer, encodings, predictions):
    all_predictions = []
    for x, prediction in enumerate(predictions[0]):
        tokens = bert_tokenizer.convert_ids_to_tokens(encodings.input_ids[x])
        token_list = tokens[
            np.argmax(predictions[0][x]) : np.argmax(predictions[1][x]) + 1
        ]
        answer = ""
        for token in token_list:
            if token.startswith("##"):
                answer += token[2:]
            else:
                answer += " " + token
        all_predictions.append(answer)
    return all_predictions


bert_qa_model = create_bert_qa_model()
# tf.keras.utils.plot_model(bert_qa_model, show_shapes=True)
bert_qa_model.summary()

predictions = bert_qa_model.predict(
    [
        train_encodings.input_ids,
        train_encodings.token_type_ids,
        train_encodings.attention_mask,
    ]
)
answers = combine_bert_subwords(bert_tokenizer, train_encodings, predictions)

for i, q in enumerate(train_question):
    print(f"Question: {q}")
    print(f"Predicted Answer: {answers[i]}")
    if ds_train_input[i]["is_impossible"]:
        print(
            f"Plausible Answer: {ds_train_input[i]['plausible_answers']['text'][0].decode('utf-8')}"
        )
    else:
        print(f"Answer: {ds_train_input[i]['answers']['text'][0].decode('utf-8')}")
    print(80 * "=")
