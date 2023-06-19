import json
import random
from pathlib import Path
from typing import Type, Union

from torch.utils.data import Dataset

from velvet.instruction_template import ImageCaptionTemplate, InstructionTemplate
from velvet.iso639_1 import iso639_1_list


class COCO(Dataset):
    def __init__(
        self,
        image_root_dir: Path,
        metadata_root_dir: Path,
        iso639_1_code: str,
        shuffle_seed: int = None,
        template_class: Type[
            Union[InstructionTemplate, ImageCaptionTemplate]
        ] = InstructionTemplate,
    ) -> None:
        """
        Parameters
        ----------
        image_root_dir : Path
            Directory where contain coco train2017/
        metadata_root_dir : Path
            Directory where contain coco-2017-vi en/, vi/...
        iso639_1_code : str
            Language code following ISO 639-1
        shuffle_seed : int, optional
            if shuffle_seed is None, don't shuffle dataset else shuffle according to seed value, by default None
        template_class : Type[ Union[InstructionTemplate, ImageCaptionTemplate] ], optional
            Template to create instruction response pair, by default InstructionTemplate
        """
        assert any(
            [iso639_1_code == i.part1 for i in iso639_1_list]
        ), f"{iso639_1_code} is not a valid ISO 639-1 code"
        self.iso639_1_code = iso639_1_code

        self.image_root_dir = image_root_dir
        self.image_split_dir = image_root_dir.joinpath("train2017")

        self.metadata_root_dir = metadata_root_dir
        if iso639_1_code == "vi":
            self.json_file = metadata_root_dir.joinpath(
                f"{iso639_1_code}/captions_train2017_trans_plus.json"
            )
        else:
            self.json_file = metadata_root_dir.joinpath(
                f"{iso639_1_code}/captions_train2017.json"
            )
        assert (
            self.json_file.is_file()
        ), f"{iso639_1_code}/captions_train2017*.json file(s) be not exist"

        self.dataset = json.load(open(self.json_file, "r", encoding="utf-8"))[
            "annotations"
        ]
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
            _description_
        """
        data = self.dataset[index]
        image_file = self.image_split_dir.joinpath(
            str(data["image_id"]).zfill(12) + ".jpg"
        )
        data["image_file"] = image_file

        instruction, response = self.template_class.make_instruction_response_pair(
            caption=data["caption"],
            language=self.iso639_1_code,
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data
