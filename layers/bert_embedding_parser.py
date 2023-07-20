import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("INFO")

import models.data_load as data_load


def generate_bert_embeddings(
    bert_model: tf.keras.Model,
    train_ids,
    train_tokens,
    train_masks,
    batch_size: int,
    idx: int,
):
    """
    load the hidden layers from the model and write the embeddings into files.
    returns the embeddings with (batch_size, 386, 1024, 25) shape
    """

    embeddings = np.zeros((batch_size, 386, 1024, 25), dtype=np.float16)
    e = bert_model.predict(
        [
            train_ids[idx : idx + batch_size],
            train_masks[idx : idx + batch_size],
            train_tokens[idx : idx + batch_size],
        ]
    )[0]

    for j in range(25):
        embeddings[:, :, :, j] = e[j]

    return embeddings


def load_bert_embeddings(model, batch_size):
    """Loads the bert embeddings in by the batches.
    Since BERT embeddings with all layers are too large to fit into memory (~5TB),
    we decided to generate the embeddings in batches and load them in by batches.
    """
    (
        train_ids,
        train_tokens,
        train_masks,
        train_impossible,
        train_start_positions,
        train_end_positions,
        qas_id,
    ) = data_load.load_train()

    labels = np.vstack([train_start_positions, train_end_positions]).T
    # (130319, 2)
    # transformed for the ease of indexing

    indices = list(range(len(labels) // batch_size))
    np.random.shuffle(indices)

    offset = 0

    while True:
        darray = np.zeros((batch_size, 386, 1024, 25), dtype=np.float16)
        train_labels = [np.zeros(batch_size), np.zeros(batch_size)]

        if offset > len(indices) - 1:
            offset = 0
        next_offset = offset + 1

        index = indices[offset]
        data = generate_bert_embeddings(
            model, train_ids, train_tokens, train_masks, batch_size, index
        )

        darray[0:batch_size, :, :, :] = data

        train_labels[0] = labels[index : index + batch_size].T[0]
        train_labels[1] = labels[index : index + batch_size].T[1]

        if next_offset > len(indices) - 1:
            output = darray[: int(len(index) * batch_size)]
            labs = [
                tf.convert_to_tensor(train_labels[0][: int(len(index) * batch_size)]),
                tf.convert_to_tensor(train_labels[1][: int(len(index) * batch_size)]),
            ]
            offset = 0

        else:
            output = darray
            labs = train_labels
            offset += 1
        # print(output.shape)
        yield output, labs
