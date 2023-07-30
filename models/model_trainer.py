import pickle
from pathlib import Path

import joblib
import tensorflow as tf

import models.data_load as data_load
from models.model_scorer import generate_scoring_dict

if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()


class Histories(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.start_accuracies = []
        self.end_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.start_accuracies.append(logs.get("softmax_3_final_start_acc"))
        self.end_accuracies.append(logs.get("softmax_3_final_end_acc"))


def train_model(model: tf.keras.Model, epoch_count=None, optimizer="adam", epochs=1, batch_size=16):
    """
    Compile and train the given BERT model, saving the history and generating scoring dict
    """

    if epoch_count:
        data_path = data_load.cache_path.joinpath(f"training_data_{epoch_count}.pkl").resolve()
        print(f"loading the training data from {data_path}...")
        training_data = joblib.load(str(data_path))

        input_ids = training_data["input_ids"]
        token_type_ids = training_data["token_type_ids"]
        attention_mask = training_data["attention_mask"]
        start_positions = training_data["start_positions"]
        end_positions = training_data["end_positions"]
    else:
        print("loading the training data...")
        (
            input_ids,
            token_type_ids,
            attention_mask,
            _,
            start_positions,
            end_positions,
            _,
        ) = data_load.load_train()

    # compile the model

    model.compile(
        optimizer=optimizer,
        loss=[
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        ],
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="final_start_acc"),
            tf.keras.metrics.SparseCategoricalAccuracy(name="final_end_acc"),
        ],
    )

    # model.summary()
    model_path = script_path.joinpath(model.name)
    checkpoint_fullpath = model_path.joinpath("training_checkpoints/ckpt_{epoch:04d}.ckpt")
    histories = Histories()

    history = model.fit(
        [input_ids, token_type_ids, attention_mask],
        [
            start_positions,
            end_positions,
        ],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_fullpath,
                verbose=1,
                save_weights_only=True,
                save_freq="epoch",
            ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor="val_loss",
            #     mode="min",
            #     verbose=1,
            #     patience=20,
            #     min_delta=0.0001,
            #     restore_best_weights=True,
            # ),
            histories,
        ],
    )

    print("Save history...")
    joblib.dump(
        (histories.losses, histories.start_accuracies, histories.end_accuracies),
        str(model_path.joinpath(f"{model.name}_histories_by_batch.pkl")),
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    joblib.dump(
        history.history,
        str(model_path.joinpath(f"{model.name}_history.pkl")),
        compress=False,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    generate_scoring_dict(model)
