import json
import random
from pathlib import Path
from typing import Type, Union

from torch.utils.data import Dataset

from velvet.instruction_template import (
    InstructionTemplate,
    ReverseVisualQuestionAnswerTemplate,
    VisualQuestionAnswerTemplate,
)
from velvet.iso639_1 import iso639_1_list


class OKVQA(Dataset):
    def __init__(
        self,
        image_root_dir: Path,
        metadata_root_dir: Path,
        iso639_1_code: str,
        shuffle_seed: int = None,
        template_class: Type[
            Union[
                InstructionTemplate,
                VisualQuestionAnswerTemplate,
                ReverseVisualQuestionAnswerTemplate,
            ]
        ] = InstructionTemplate,
    ) -> None:
        """
        Parameters
        ----------
        image_root_dir : Path
            Directory where contain coco train2017/
        metadata_root_dir : Path
            Directory where contain OK-VQA-multilang en/, vi/...
        iso639_1_code : str
            Language code following ISO 639-1
        shuffle_seed : int, optional
            if shuffle_seed is None, don't shuffle dataset else shuffle according to seed value, by default None
        template_class : Type[ Union[ InstructionTemplate, VisualQuestionAnswerTemplate, ReverseVisualQuestionAnswerTemplate, ] ], optional
            Template to create instruction response pair, by default InstructionTemplate
        """
        assert any(
            [iso639_1_code == i.part1 for i in iso639_1_list]
        ), f"{iso639_1_code} is not a valid ISO 639-1 code"
        self.iso639_1_code = iso639_1_code

        self.image_root_dir = image_root_dir
        self.image_split_dir = image_root_dir.joinpath("train2017")

        self.metadata_root_dir = metadata_root_dir
        self.question_json_file = metadata_root_dir.joinpath(
            f"{iso639_1_code}/OpenEnded_mscoco_train2014_questions.json"
        )
        assert (
            self.question_json_file.is_file()
        ), f"{iso639_1_code}/OpenEnded_mscoco_train2014_questions.json does not exist"

        self.answer_json_file = metadata_root_dir.joinpath(
            f"{iso639_1_code}/mscoco_train2014_annotations.json"
        )
        assert (
            self.answer_json_file.is_file()
        ), f"{iso639_1_code}/mscoco_train2014_annotations.json does not exist"

        self.template_class = template_class

        self.__make_dataset()
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

    def __make_dataset(self):
        """For the love of the gods, don't call me out publicly"""
        self.dataset = []

        question_list: list = json.load(
            open(self.question_json_file, "r", encoding="utf-8")
        )["questions"]
        answer_list: list = json.load(
            open(self.answer_json_file, "r", encoding="utf-8")
        )["annotations"]

        question_list.sort(key=lambda d: d["question_id"])
        answer_list.sort(key=lambda d: d["question_id"])

        for q, a in zip(question_list, answer_list):
            assert q["question_id"] == a["question_id"]
            assert q["image_id"] == a["image_id"]
            data = {}
            data.update(q)
            data.update(a)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        """
        Parameters
        ----------
        index : _type_
            index in the json file

        Returns
        -------
        dict
            See https://visualqa.org/download.html
        """
        data = self.dataset[index]
        image_file = self.image_split_dir.joinpath(
            str(data["image_id"]).zfill(12) + ".jpg"
        )
        data["image_file"] = image_file

        none_empty_answer = "None"
        for a in data["answers"]:
            if a["answer"] != "":
                none_empty_answer = a["answer"]
                break

        instruction, response = self.template_class.make_instruction_response_pair(
            question=data["question"],
            answer=none_empty_answer,
            language=self.iso639_1_code,
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data
