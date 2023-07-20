from pathlib import Path

import joblib

check_point_path = "./results/bert-large-uncased/training_checkpoints/"
data_path = Path("./data/")
cache_path = Path("./")


def get_checkpoint_path(epoch_number):
    return check_point_path + f"ckpt_000{epoch_number}.ckpt"


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
