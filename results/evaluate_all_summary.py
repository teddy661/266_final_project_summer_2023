import json
import os
from pathlib import Path

import pandas as pd


def extract_metrics():
    """
    loop through all json in the current directory, and extract
        "exact",
        "f1",
        "HasAns_exact"
        "HasAns_f1"
        "NoAns_exact"
        "NoAns_f1"
    into a dataframe, with the model name (json file name) as the index
    """

    # Get the current directory
    current_directory = Path(__file__).parent.absolute()
    os.chdir(current_directory)

    # Create an empty list to store the extracted metrics
    metrics_data = []

    # Loop through all JSON files in the current directory
    for filename in os.listdir(current_directory):
        if filename.endswith(".json"):
            with open(filename) as file:
                data = json.load(file)

                # Extract the desired metrics from the JSON
                model_name = os.path.splitext(filename)[0]
                exact = data.get("exact")
                f1 = data.get("f1")
                has_ans_exact = data.get("HasAns_exact")
                has_ans_f1 = data.get("HasAns_f1")
                no_ans_exact = data.get("NoAns_exact")
                no_ans_f1 = data.get("NoAns_f1")

                # Append the metrics to the list
                metrics_data.append(
                    {
                        "Model Name": model_name,
                        "Exact": exact,
                        "F1": f1,
                        "HasAns_Exact": has_ans_exact,
                        "HasAns_F1": has_ans_f1,
                        "NoAns_Exact": no_ans_exact,
                        "NoAns_F1": no_ans_f1,
                    }
                )

    # Create a DataFrame from the extracted metrics
    df = pd.DataFrame(metrics_data)

    # Set the model name as the index
    df["Epoch"] = df["Model Name"].str[-2:].astype(float)
    df["Epoch"] = df["Epoch"] / 100
    df.set_index(["Epoch", "Model Name"], inplace=True)
    df.sort_index(inplace=True)

    return df


(extract_metrics()).to_csv("metrics_partial.csv")
