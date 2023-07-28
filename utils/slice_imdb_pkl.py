import joblib
import tensorflow as tf
from pathlib import Path

if "__file__" in globals():
    script_path = Path(__file__).parent.absolute()
else:
    script_path = Path.cwd()

data_dir = script_path.joinpath("../data")
training_data_path = data_dir.joinpath("imdb_training_data.pkl")
training_data = joblib.load(training_data_path)

possible_data_percent = [20, 40, 60, 80, 100]


for percent_data in possible_data_percent:
    new_dict = {}
    end_slice = int(training_data["input_ids"].shape[0] * percent_data // 100)
    for key in training_data.keys():
        new_dict[key] = training_data[key][:end_slice]
    joblib.dump(new_dict, data_dir.joinpath(f"imdb_training_data_{percent_data}.pkl"))
