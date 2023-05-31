import json
from pathlib import Path

data_path = Path("/storage/anhvd/data/EVJVQA/evjvqa_train.json")
data_json = json.load(open(data_path))

start_idx_lookup = {"en": 0, "vi": 7204, "ja": 15524}

for i, data in enumerate(data_json["annotations"]):
    if start_idx_lookup["en"] <= i < start_idx_lookup["vi"]:
        data["language"] = "en"
    elif start_idx_lookup["vi"] <= i < start_idx_lookup["ja"]:
        data["language"] = "vi"
    elif start_idx_lookup["ja"] <= i < len(data_json["annotations"]):
        data["language"] = "ja"

save_path = Path("/storage/anhvd/data/EVJVQA/lang_evjvqa_train.json")
json.dump(
    data_json, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4
)
