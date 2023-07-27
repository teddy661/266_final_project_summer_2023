import os
import shutil
from pathlib import Path

import joblib
import tensorflow as tf
from transformers import BertTokenizer

if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
data_dir = script_path.joinpath("../data").resolve()
imdb_data_dir = data_dir.joinpath("imdb").resolve()
imdb_tar_file = imdb_data_dir.joinpath("aclImdb_v1.tar.gz").resolve()
if not imdb_tar_file.exists():
    print("Downloading IMDB dataset...")
    dataset = tf.keras.utils.get_file(
        fname=imdb_tar_file.name,
        origin=url,
        extract=True,
        cache_dir=imdb_data_dir,
        cache_subdir="",
    )
    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")
    train_dir = os.path.join(dataset_dir, "train")
    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, "unsup")
    shutil.rmtree(remove_dir)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 1
seed = 42

imdb_train_dir = imdb_data_dir.joinpath("aclImdb/train")
imdb_test_dir = imdb_data_dir.joinpath("aclImdb/test")
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    imdb_train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    imdb_train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    imdb_test_dir, batch_size=batch_size
)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
max_seq_length = 512
max_query_length = 64
doc_stride = 128

input_ids = []
token_type_ids = []
attention_masks = []
labels = []
# Oh this is so wrong, but I don't know how to do it right:
for data_element in train_ds.take(train_ds.cardinality().numpy()):
    labels.append(data_element[1][0].numpy())
    tokenizer_result = tokenizer(
        data_element[0][0].numpy().decode("utf-8"),
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
    )
    input_ids.append(tokenizer_result["input_ids"])
    token_type_ids.append(tokenizer_result["token_type_ids"])
    attention_masks.append(tokenizer_result["attention_mask"])


input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
attention_mask = tf.convert_to_tensor(attention_masks, dtype=tf.int64)
labels = tf.convert_to_tensor(labels, dtype=tf.int64)

training_dict = {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}
training_pkl_file = data_dir.joinpath("imdb_training_data.pkl").resolve()
print("Saving training data to ", training_pkl_file)
joblib.dump(training_dict, training_pkl_file)

input_ids = []
token_type_ids = []
attention_masks = []
labels = []
# Oh this is so wrong, but I don't know how to do it right:
for data_element in val_ds.take(val_ds.cardinality().numpy()):
    labels.append(data_element[1][0].numpy())
    tokenizer_result = tokenizer(
        data_element[0][0].numpy().decode("utf-8"),
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
    )
    input_ids.append(tokenizer_result["input_ids"])
    token_type_ids.append(tokenizer_result["token_type_ids"])
    attention_masks.append(tokenizer_result["attention_mask"])


input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int64)
token_type_ids = tf.convert_to_tensor(token_type_ids, dtype=tf.int64)
attention_mask = tf.convert_to_tensor(attention_masks, dtype=tf.int64)
labels = tf.convert_to_tensor(labels, dtype=tf.int64)

training_dict = {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}
validation_pkl_file = data_dir.joinpath("imdb_validation_data.pkl").resolve()
print("Saving validation data to ", validation_pkl_file)
joblib.dump(training_dict, validation_pkl_file)
