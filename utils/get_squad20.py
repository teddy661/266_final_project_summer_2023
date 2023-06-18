import json
import sys

import requests
from transformers.data.processors.squad import squad_convert_examples_to_features

SQUAD20_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
SQUAD20_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD20_EVALUATE_URL = "https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/"
SQUAD20_SAMPLE_PREDICTION_URL = (
    "https://worksheets.codalab.org/bundles/0x8731effab84f41b7b874a070e40f61e2/"
)


sq_trn_req = requests.get(SQUAD20_TRAIN_URL, stream=True, allow_redirects=True)
sq_trn_req.raw.decode_content = True
squad_train_json = json.loads(sq_trn_req.raw.read().decode("utf-8"))

sq_dev_req = requests.get(SQUAD20_DEV_URL, stream=True, allow_redirects=True)
sq_dev_req.raw.decode_content = True
squad_dev_json = json.loads(sq_dev_req.raw.read().decode("utf-8"))
