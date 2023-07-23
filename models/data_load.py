from pathlib import Path

import joblib

cache_path = Path("./data/")


def get_checkpoint_path(epoch_number):
    if epoch_number < 10:
        return (
            Path(__file__)
            .parent.parent.joinpath(
                f"models/bert-large-uncased/training_checkpoints/ckpt_000{epoch_number}.ckpt"
            )
            .resolve()
            .__str__()
        )
    else:
        return (
            Path(__file__)
            .parent.parent.joinpath(
                f"models/bert-large-uncased_{epoch_number}/training_checkpoints/ckpt_0001.ckpt"
            )
            .resolve()
            .__str__()
        )


def load_dev():
    return load_data((cache_path.joinpath("dev_cached_feature.pkl")).resolve())


def load_train(cap=None):
    (
        input_ids,
        token_type_ids,
        attention_mask,
        impossible,
        start_positions,
        end_positions,
        qas_id,
    ) = load_data((cache_path.joinpath("train_cached_feature.pkl")).resolve())

    if cap:
        return (
            input_ids[:cap],
            token_type_ids[:cap],
            attention_mask[:cap],
            impossible[:cap],
            start_positions[:cap],
            end_positions[:cap],
            qas_id[:cap],
        )
    else:
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            impossible,
            start_positions,
            end_positions,
            qas_id,
        )


def load_data(cached_data_path: Path):
    """
    load from cached files directly instead of using tf.data.Dataset.load()
    """

    if cached_data_path.exists():
        return joblib.load(cached_data_path, "r")
    else:
        raise FileNotFoundError(
            f"File not found: {cached_data_path}... Run the create_squadv2_datasets.py script first..."
        )
