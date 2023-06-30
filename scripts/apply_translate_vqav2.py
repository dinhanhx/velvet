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


def apply_translate_vqav2(root_path: Path, iso639_1_code: str):
    en_question_json_file = root_path.joinpath(
        "en/v2_OpenEnded_mscoco_train2014_questions.json"
    )
    en_answer_json_file = root_path.joinpath(
        "en/v2_OpenEnded_mscoco_train2014_answers.json"
    )

    en_question_list_path = root_path.joinpath("en/question_list.txt")
    en_answer_list_path = root_path.joinpath("en/answer_list.txt")

    question_map_path = root_path.joinpath(f"{iso639_1_code}/question_list.jsonl")
    answer_map_path = root_path.joinpath(f"{iso639_1_code}/answer_list.jsonl")

    question_map = create_map(en_question_list_path, question_map_path)
    answer_map = create_map(en_answer_list_path, answer_map_path)

    print(len(question_map), len(answer_map))

    en_question_json = json.load(open(en_question_json_file, "r", encoding="utf-8"))
    en_answer_json = json.load(open(en_answer_json_file, "r", encoding="utf-8"))

    other_question_json = deepcopy(en_question_json)
    other_answer_json = deepcopy(en_answer_json)

    for data, en_data in zip(
        other_question_json["questions"], en_question_json["questions"]
    ):
        data["question"] = question_map[en_data["question"]]

    other_question_path = root_path.joinpath(
        f"{iso639_1_code}/v2_OpenEnded_mscoco_train2014_questions.json"
    )
    with open(other_question_path, "w", encoding="utf-8") as fp:
        json.dump(other_question_json, fp, ensure_ascii=False)

    for data, en_data in zip(
        other_answer_json["annotations"], en_answer_json["annotations"]
    ):
        data["multiple_choice_answer"] = answer_map[en_data["multiple_choice_answer"]]

    other_answer_path = root_path.joinpath(
        f"{iso639_1_code}/v2_OpenEnded_mscoco_train2014_answers.json"
    )
    with open(other_answer_path, "w", encoding="utf-8") as fp:
        json.dump(other_answer_json, fp, ensure_ascii=False)

    return other_question_path, other_answer_path


apply_translate_vqav2(Path(config_data_dir["vqav2"]["metadata_root_dir"]), "vi")
