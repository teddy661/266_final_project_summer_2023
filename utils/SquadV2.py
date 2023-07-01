import tensorflow as tf
import tensorflow_datasets as tfds


class SquadV2:
    def __init__(self):
        self.train = None
        self.validation = None
        self.info = None

    def load_data(self):
        (ds_train, ds_validation), self.info = tfds.load(
            "squad/v2.0",
            split=["train", "validation"],
            shuffle_files=False,
            with_info=True,
        )

        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(self.info.splits["train"].num_examples)
        self.train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_validation = ds_validation.cache()
        ds_validation = ds_validation.shuffle(self.info.splits["train"].num_examples)
        self.validation = ds_validation.prefetch(tf.data.AUTOTUNE)

    def get_raw_train_data(self):
        return self.train

    def get_raw_validation_data(self):
        return self.validation

    def get_train_data(self, num_samples=None):
        if num_samples is None:
            num_samples = self.train.cardinality().numpy()
        return [x for x in tfds.as_numpy(self.train.take(num_samples))]

    def get_validation_data(self, num_samples=None):
        if num_samples is None:
            num_samples = self.validation.cardinality().numpy()
        return [x for x in tfds.as_numpy(self.validation.take(num_samples))]

    def get_info(self):
        return self.info
