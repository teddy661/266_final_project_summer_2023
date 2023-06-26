import tensorflow_datasets as tfds

(squadv2_train, squadv2_validation), squadv2_info = tfds.load("squad/v2.0", split=["train", "validation"], shuffle_files=False, with_info=True)

squadv2_train_input = [x for x in tfds.as_numpy(squadv2_train)]
squadv2_validation_input = [x for x in tfds.as_numpy(squadv2_validation)]
