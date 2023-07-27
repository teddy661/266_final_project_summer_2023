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
if not data_dir.exists():
    data_dir.mkdir(parents=True)
imdb_data_dir = data_dir.joinpath("imdb").resolve()
if not imdb_data_dir.exists():
    imdb_data_dir.mkdir(parents=True)
imdb_tar_file = imdb_data_dir.joinpath("aclImdb_v1.tar.gz").resolve()
if not imdb_data_dir.exists():
    imdb_data_dir.mkdir(parents=True)
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
batch_size = 32
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

labels = []
text_list = []
# Oh this is so wrong, but I don't know how to do it right:
# for data_element in train_ds.take(train_ds.cardinality().numpy()):
for text_element, label_element in train_ds.take(
    train_ds.cardinality().numpy()
).as_numpy_iterator():
    for raw_bytes in text_element:
        text_list.append(raw_bytes.decode("utf-8"))
    labels.extend(label_element)

labels = tf.convert_to_tensor(labels, dtype=tf.int64)

tokenizer_result = tokenizer(
    text_list,
    max_length=max_seq_length,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
)

training_dict = {
    "input_ids": tokenizer_result["input_ids"],
    "token_type_ids": tokenizer_result["token_type_ids"],
    "attention_mask": tokenizer_result["attention_mask"],
    "labels": labels,
}
training_pkl_file = data_dir.joinpath("imdb_training_data.pkl").resolve()
print("Saving training data to ", training_pkl_file)
joblib.dump(training_dict, training_pkl_file)

labels = []
text_list = []
for text_element, label_element in val_ds.take(
    val_ds.cardinality().numpy()
).as_numpy_iterator():
    for raw_bytes in text_element:
        text_list.append(raw_bytes.decode("utf-8"))
    labels.extend(label_element)

labels = tf.convert_to_tensor(labels, dtype=tf.int64)

tokenizer_result = tokenizer(
    text_list,
    max_length=max_seq_length,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
)

training_dict = {
    "input_ids": tokenizer_result["input_ids"],
    "token_type_ids": tokenizer_result["token_type_ids"],
    "attention_mask": tokenizer_result["attention_mask"],
    "labels": labels,
}
validation_pkl_file = data_dir.joinpath("imdb_validation_data.pkl").resolve()
print("Saving validation data to ", validation_pkl_file)
joblib.dump(training_dict, validation_pkl_file)
