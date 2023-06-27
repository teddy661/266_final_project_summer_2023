import tensorflow as tf
import tensorflow_datasets as tfds

(squadv2_train, squadv2_validation), squadv2_info = tfds.load(
    "squad/v2.0", split=["train", "validation"], shuffle_files=False, with_info=True
)

input = squadv2_train.map(
    lambda x: {
        "question": x["question"],
        "context": x["context"],
        "answers": x["answers"],
    },
    num_parallel_calls=tf.data.AUTOTUNE,
)

input = input.cache()
input = input.prefetch(tf.data.AUTOTUNE)
