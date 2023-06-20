import json
from copy import deepcopy
from pathlib import Path

import toml

config_data_dir = toml.load(open("configs/data_dir.toml", "r"))


def create_map(en_list_path: Path, other_jsonl_path: Path):
    with open(en_list_path, "r", encoding="utf-8") as fp:
        en_list = []
        for s in fp:
            en_list.append(s.rstrip("\n"))

    with open(other_jsonl_path, "r", encoding="utf-8") as fp:
        other_list = []
        for l in fp:
            other_list.append(list(json.loads(l).values())[0])

    return dict(zip(en_list, other_list))


def apply_translate_textvqa(root_path: Path, iso639_1_code: str, split: str):
    def get_first_non_empty_answer(answers: list):
        for a in answers:
            if a != "":
                return a
        return "None"

    en_json_path = root_path.joinpath(f"en/TextVQA_0.5.1_{split}.json")
    en_question_list = root_path.joinpath(f"en/{split}_question_list.txt")
    en_answer_list = root_path.joinpath(f"en/{split}_answer_list.txt")
    question_map_path = root_path.joinpath(
        f"{iso639_1_code}/{split}_question_list.jsonl"
    )
    answer_map_path = root_path.joinpath(f"{iso639_1_code}/{split}_answer_list.jsonl")

    en_json = json.load(open(en_json_path, "r", encoding="utf-8"))
    other_json = deepcopy(en_json)

    question_map = create_map(en_question_list, question_map_path)
    answer_map = create_map(en_answer_list, answer_map_path)

    print(len(question_map), len(answer_map))

    for data, en_data in zip(other_json["data"], en_json["data"]):
        data["question"] = question_map[en_data["question"]]
        en_answer = get_first_non_empty_answer(en_data["answers"])
        data["answers"] = [answer_map[en_answer]]

    other_json_path = root_path.joinpath(f"{iso639_1_code}/TextVQA_0.5.1_{split}.json")
    with open(other_json_path, "w", encoding="utf-8") as fp:
        json.dump(other_json, fp, ensure_ascii=False)
    return other_json_path


print(apply_translate_textvqa(Path(config_data_dir['textvqa']['metadata_root_dir']), "vi", "train"))
print(apply_translate_textvqa(Path(config_data_dir['textvqa']['metadata_root_dir']), "vi", "val"))
