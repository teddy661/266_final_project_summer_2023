import h5py
import numpy as np
from transformers import BertConfig, BertTokenizer, TFBertModel

import data.data_load as data_load

data_dir = "../data/bert_embeddings/"


def write_file(directory, idx, embeddings):
    """
    Write the embeddings to a h5 file
    """
    with h5py.File(directory + str(idx) + ".h5", "w") as f:
        f.create_dataset("hidden_state_activations", data=embeddings)


def generate_bert_embeddings(model):
    """
    load the hidden layers from the model and write the embeddings into files
    """

    (
        train_ids,
        train_tokens,
        train_masks,
        impossible,
        start_positions,
        end_positions,
        qas_id,
    ) = data_load.load_train()

    embeddings = np.zeros((8, 386, 1024, 25), dtype=np.float16)
    for i in range(12697, 16489):
        e, _ = model.model.predict(
            [
                train_ids[i * 8 : (i + 1) * 8],
                train_masks[i * 8 : (i + 1) * 8],
                train_tokens[i * 8 : (i + 1) * 8],
            ]
        )
        for j in range(25):
            embeddings[:, :, :, j] = e[j]

        if e[0].shape[0] == 8:
            write_file(data_dir, i * 8, embeddings)
        else:
            write_file(data_dir, i * 8, embeddings[: e[0].shape[0]])
        if not i % 1000:
            print(i)


bert_config = BertConfig.from_pretrained(
    "bert-large-uncased",
    output_hidden_states=True,
)

bert_model = TFBertModel.from_pretrained("bert-large-uncased", config=bert_config)
generate_bert_embeddings(bert_model)


def load_bert_embeddings(
    labels,
    indices,
    batch_size=32,
    file_size=8,
    truncate_data=False,
):
    """Loads the bert embeddings in by the batches.
    Since BERT embeddings with all layers are too large to fit into memory (~5TB),
        we must load them from disk so it is an expensive process
    """

    np.random.shuffle(indices)

    # Use a smaller fraction of the data by truncating randomized indices to specified number
    if truncate_data:
        indices = indices[:truncate_data]

    offset = 0
    splits = int(batch_size / file_size)

    while True:
        darray = np.zeros((batch_size, 386, 1024, 25), dtype=np.float16)
        train_labels = [np.zeros(batch_size), np.zeros(batch_size)]

        if offset * splits > len(indices) - 1:
            offset = 0

        # Take a data slice
        data_slice = indices[offset * splits : (offset + 1) * splits]
        # darray[:] = 0

        for i, idx in enumerate(data_slice):
            with h5py.File(data_dir + "%d.h5" % (idx), "r") as f:
                data = np.array(f["hidden_state_activations"], dtype=np.float16)

            darray[i * file_size : (i + 1) * file_size, :, :, :] = data

            train_labels[0][i * file_size : (i + 1) * file_size] = labels[
                idx : idx + file_size
            ].T[0]
            train_labels[1][i * file_size : (i + 1) * file_size] = labels[
                idx : idx + file_size
            ].T[1]

        if (offset + 1) * splits > len(indices) - 1:
            output = darray[: int(len(data_slice) * file_size)]
            labs = [
                train_labels[0][: int(len(data_slice) * file_size)],
                train_labels[1][: int(len(data_slice) * file_size)],
            ]
            offset = 0

        else:
            output = darray
            labs = train_labels
            offset += 1
        # print(output.shape)
        yield output, labs
