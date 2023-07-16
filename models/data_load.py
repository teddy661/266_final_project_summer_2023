import pickle
from pathlib import Path

import joblib


def load_dev():
    return load_data((Path(__file__).parent.joinpath("../dev_cached_feature.pkl")).resolve())


def load_train():
    return load_data((Path(__file__).parent.joinpath("../train_cached_feature.pkl")).resolve())


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
