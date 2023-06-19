import json
import random
from pathlib import Path
from typing import Type, Union

from torch.utils.data import Dataset

from velvet.instruction_template import (
    ImageCaptionTemplate,
    InstructionTemplate,
    ReverseVisualQuestionAnswerTemplate,
    VisualQuestionAnswerTemplate,
)
from velvet.iso639_1 import iso639_1_list


class EVJVQA(Dataset):
    def __init__(
        self,
        root_dir: Path,
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
        root_dir : Path
            Directory where contain EVJVQA train-images/, lang_evjvqa_train.json
        shuffle_seed : int, optional
            if shuffle_seed is None, don't shuffle dataset else shuffle according to seed value
        template_class : Type[Union[ InstructionTemplate, VisualQuestionAnswerTemplate, ReverseVisualQuestionAnswerTemplate, ]], optional
            Template to create instruction response pair, by default InstructionTemplate
        """

        self.root_dir = root_dir
        self.img_dir = root_dir.joinpath("train-images")
        self.json_file = root_dir.joinpath("lang_evjvqa_train.json")

        self.dataset = json.load(open(self.json_file, "r", encoding="utf-8"))[
            "annotations"
        ]
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

        self.template_class = template_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """
        Parameters
        ----------
        index : int
            index in the json file

        Returns
        -------
        dict
            {'id': 0,
            'image_id': 2301,
            'question': 'what color is the shirt of the girl wearing glasses?',
            'answer': 'the girl wearing glasses wears a red shirt',
            'language': 'en',
            'image_file': PosixPath('/storage/anhvd/data/EVJVQA/train-images/00000002301.jpg')}
        """
        data = self.dataset[index]
        fmt_image_id = str(data["image_id"]).zfill(11)
        data["image_file"] = list(self.img_dir.glob(f"{fmt_image_id}.*"))[0]

        instruction, response = self.template_class.make_instruction_response_pair(
            question=data["question"],
            answer=data["answer"],
            language=data["language"],
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data


class CCSBUAlign(Dataset):
    def __init__(
        self,
        root_dir: Path,
        iso639_1_code: str,
        shuffle_seed: int = None,
        template_class: Type[
            Union[InstructionTemplate, ImageCaptionTemplate]
        ] = InstructionTemplate,
    ) -> None:
        """
        Parameters
        ----------
        root_dir : Path
            Directory where contain cc_sbu_align image/, en/, vi/...
        iso639_1_code : str
            Language code following ISO 639-1
        shuffle_seed : int, optional
            if shuffle_seed is None, don't shuffle dataset else shuffle according to seed value
        template_class : Type[ Union[InstructionTemplate, ImageCaptionTemplate] ], optional
            Template to create instruction response pair, by default InstructionTemplate
        """
        assert any(
            [iso639_1_code == i.part1 for i in iso639_1_list]
        ), f"{iso639_1_code} is not a valid ISO 639-1 code"
        self.iso639_1_code = iso639_1_code

        self.root_dir = root_dir
        self.img_dir = root_dir.joinpath("image")

        assert root_dir.joinpath(
            f"{iso639_1_code}/filter_cap.json"
        ).is_file(), f"{iso639_1_code}/filter_cap.json does not exist"
        self.json_file = root_dir.joinpath(f"{iso639_1_code}/filter_cap.json")

        self.dataset = json.load(open(self.json_file, "r", encoding="utf-8"))[
            "annotations"
        ]
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

        self.template_class = template_class

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """
        Parameters
        ----------
        index : int
            index in the json file

        Returns
        -------
        dict
            {'image_id': '3108',
            'caption': 'The image shows two toy cars, one red and one yellow, with different hood ornaments and ..."',
            'image_file': PosixPath('/mnt/storage/data/cc_sbu_align_multilang/image/3108.jpg')}
        """
        data = self.dataset[index]
        image_id = data["image_id"]
        data["image_file"] = list(self.img_dir.glob(f"{image_id}.*"))[0]

        instruction, response = self.template_class.make_instruction_response_pair(
            caption=data["caption"],
            language=self.iso639_1_code,
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data
