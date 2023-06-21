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


class TextVQA(Dataset):
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
            Directory where contain OpenImages train_images/
        metadata_root_dir : Path
            Directory where contain TextVQA-vi en/, vi/...
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
        self.image_dir = image_root_dir.joinpath("train_images")

        self.meta_root_dir = metadata_root_dir
        self.json_file = metadata_root_dir.joinpath(
            f"{iso639_1_code}/TextVQA_0.5.1_train.json"
        )
        assert (
            self.json_file.is_file()
        ), f"{iso639_1_code}/TextVQA_0.5.1_train.json does not exist"

        self.dataset = json.load(open(self.json_file, "r", encoding="utf-8"))["data"]
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

        self.template_class = template_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        """
        Parameters
        ----------
        index : int
            index in the json file

        Returns
        -------
        dict
            See https://textvqa.org/dataset/#:~:text=suggestions%20and%20feedback.-,Description,-TextVQA%20JSON%20files
            and https://huggingface.co/datasets/dinhanhx/TextVQA-vi
        """
        data = self.dataset[index]
        image_file = self.image_dir.joinpath(data["image_id"] + ".jpg")
        data["image_file"] = image_file

        none_empty_answer = "None"
        for a in data["answers"]:
            if a != "":
                none_empty_answer = a
                break

        instruction, response = self.template_class.make_instruction_response_pair(
            question=data["question"],
            answer=none_empty_answer,
            language=self.iso639_1_code,
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data
