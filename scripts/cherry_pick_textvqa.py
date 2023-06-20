import json
from pathlib import Path

root_path = Path("/mnt/storage/data/TextVQA-vi")
en_json_path = root_path.joinpath("en/TextVQA_0.5.1_train.json")
en_json = json.load(open(en_json_path, "r", encoding="utf-8"))

en_json["data"][15799]["answers"][0] = "ezekiel 37:4-6"

with open(en_json_path, "w", encoding="utf-8") as fp:
    json.dump(en_json, fp, ensure_ascii=False)
