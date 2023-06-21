import random
from pathlib import Path
from typing import List, Type, TypedDict, Union

import polars as pl
from torch.utils.data import Dataset

from velvet.instruction_template import ImageCaptionTemplate, InstructionTemplate
from velvet.iso639_1 import iso639_1_list


class DataPointType(TypedDict):
    image_file: Path
    id: int
    caption: str


class GCC(Dataset):
    def __init__(
        self,
        gcc_vi_dir: Path,
        llava_gcc_dir: Path,
        iso639_1_code: str,
        shuffle_seed: int = None,
        template_class: Type[
            Union[InstructionTemplate, ImageCaptionTemplate]
        ] = InstructionTemplate,
    ) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        gcc_vi_dir : Path
            Directory where contain gcc-vi en/, vi/...
        llava_gcc_dir : Path
            Directory where contain LLaVA-CC3M-Pretrain-595K images.zip, images/...
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

        self.gcc_vi_dir = gcc_vi_dir
        self.caption_tsv_path = gcc_vi_dir.joinpath(f"{iso639_1_code}/train.tsv")
        assert (
            self.caption_tsv_path.is_file()
        ), f"{iso639_1_code}/train.tsv does not exist"

        self.llava_gcc_dir = llava_gcc_dir
        self.image_dir = llava_gcc_dir.joinpath("images")
        assert (
            self.image_dir.is_dir()
        ), "images/ where images.zip of liuhaotian/LLaVA-CC3M-Pretrain-595K is unzipped to does not exist"

        self.template_class = template_class

        self.__make_dataset()
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

    def __make_dataset(self):
        """For the love of the gods, don't call me out publicly"""
        self.dataset: List[DataPointType] = []

        image_file_list = list(Path(self.image_dir).glob("*"))
        caption_df = pl.read_csv(
            self.caption_tsv_path,
            has_header=False,
            new_columns=["caption", "url"],
            separator="\t",
        )

        for i in image_file_list:
            index = int(i.stem.split("_")[-1])
            caption = caption_df.row(index)[0]
            self.dataset.append({"image_file": i, "id": index, "caption": caption})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Union[DataPointType, dict]:
        """
        Parameters
        ----------
        index : int
            You came here to expect description, but it was me, no documentation

        Returns
        -------
        Union[DataPointType, dict]
            See class DataPointType in this file
        """
        data = self.dataset[index]

        instruction, response = self.template_class.make_instruction_response_pair(
            caption=data["caption"],
            language=self.iso639_1_code,
        )
        data["_instruction_"] = instruction
        data["_response_"] = response
        return data
