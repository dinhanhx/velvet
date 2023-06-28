from pathlib import Path
from typing import Union

import toml
from torch.utils.data import ConcatDataset, Dataset

from velvet.instruction_template import (
    ImageCaptionTemplate,
    VisualQuestionAnswerTemplate,
)

from .cc_sbu_align import CCSBUAlign
from .coco import COCO
from .config_data_dir_dataclass import ConfigDataDirDataclass
from .evjvqa import EVJVQA
from .gcc import GCC
from .okvqa import OKVQA
from .textcaps import TextCaps
from .textvqa import TextVQA
from .vqav2 import VQAv2

UnionVelvetDataset = Union[
    Dataset, CCSBUAlign, COCO, EVJVQA, TextCaps, TextVQA, VQAv2, OKVQA, GCC
]


def create_all_dataset_list(config_data_dir_toml_file: str, seed: int = 1312) -> list:
    """Create a list of all datasets

    This is a kinda unstable function when comes to extensibility.
    One is advised to read the whole function before using.

    Parameters
    ----------
    config_data_dir_toml_file : str
        Path in form of str points to data_dir.toml
    seed : int
        Random seed, by default 1312

    Returns
    -------
    list
        List of dict. Each dict contain:
            - `d_name`: a valid dataset in toml file
            - `d_lang`: a valid iso639-1 code
            - `d_object`: a object of dataset class
    """
    config_data_dir = toml.load(open(config_data_dir_toml_file, "r"))

    ic_dataset_names = ["coco", "textcaps"]
    ic_dataset_classes = [COCO, TextCaps]

    vqa_dataset_names = ["okvqa", "textvqa", "vqav2"]
    vqa_dataset_classes = [OKVQA, TextVQA, VQAv2]

    mixed_lang = "mixed"
    langs = ["en", "vi"]

    dataset_list = []

    dataset_list.append(
        {
            "d_name": "evjvqa",
            "d_lang": mixed_lang,
            "d_object": EVJVQA(
                Path(config_data_dir["evjvqa"]["root_dir"]),
                seed,
                VisualQuestionAnswerTemplate,
            ),
        }
    )

    dataset_list = dataset_list + [
        {
            "d_name": "gcc",
            "d_lang": lang,
            "d_object": GCC(
                Path(config_data_dir["gcc"]["gcc_vi_dir"]),
                Path(config_data_dir["gcc"]["llava_gcc_dir"]),
                lang,
                seed,
                ImageCaptionTemplate,
            ),
        }
        for lang in langs
    ]

    dataset_list = dataset_list + [
        {
            "d_name": "cc_sbu_align",
            "d_lang": lang,
            "d_object": CCSBUAlign(
                Path(config_data_dir["cc_sbu_align"]), lang, seed, ImageCaptionTemplate
            ),
        }
        for lang in langs
    ]

    for d_name, d_class in zip(ic_dataset_names, ic_dataset_classes):
        for lang in langs:
            d_object = d_class(
                Path(config_data_dir[d_name]["image_root_dir"]),
                Path(config_data_dir[d_name]["metadata_root_dir"]),
                lang,
                seed,
                ImageCaptionTemplate,
            )
            dataset_list.append(
                {"d_name": d_name, "d_lang": lang, "d_object": d_object}
            )

    for d_name, d_class in zip(vqa_dataset_names, vqa_dataset_classes):
        for lang in langs:
            d_object = d_class(
                Path(config_data_dir[d_name]["image_root_dir"]),
                Path(config_data_dir[d_name]["metadata_root_dir"]),
                lang,
                seed,
                VisualQuestionAnswerTemplate,
            )
            dataset_list.append(
                {"d_name": d_name, "d_lang": lang, "d_object": d_object}
            )

    return dataset_list


def filter_dataset_list(dataset_list: list, ignore_name: list) -> list:
    """Remove dataset from a list

    This is a kinda unstable function when comes to extensibility.
    One is advised to read the whole function before using.

    Parameters
    ----------
    dataset_list : list
        the list of all datasets that is returned by `create_all_dataset_list()`
    ignore_name : list
        List of valid dataset in toml file

    Returns
    -------
    list
        List of dict. Each dict contain:
            - `d_name`: a valid dataset in toml file
            - `d_lang`: a valid iso639-1 code
            - `d_object`: a object of dataset class
    """
    new_dataset_list = []
    for i in dataset_list:
        if i["d_name"] not in ignore_name:
            new_dataset_list.append(i)
    return new_dataset_list


def order_dataset_list(
    dataset_list: list,
    order_name: Union[list, None] = None,
    order_lang: Union[list, None] = None,
) -> list:
    """Order the list of all datasets that is the output of `create_all_dataset_list()`

    This is a kinda unstable function when comes to extensibility.
    One is advised to read the whole function before using.

    Parameters
    ----------
    dataset_list : list
        the list of all datasets that is returned by `create_all_dataset_list()`
    order_name : Union[list, None], optional
        List of valid dataset in toml file, by default ["okvqa", "textcaps", "textvqa", "coco", "vqav2", "gcc", "evjvqa"]
    order_lang : Union[list, None], optional
        List of valid iso639-1 code, by default ["en", "vi", "mixed"]

    Returns
    -------
    list
        List of dict. Each dict contain:
            - `d_name`: a valid dataset in toml file
            - `d_lang`: a valid iso639-1 code
            - `d_object`: a object of dataset class
    """
    if order_name is None:
        order_name = [
            "okvqa",
            "textcaps",
            "textvqa",
            "coco",
            "vqav2",
            "gcc",
            "evjvqa",
        ]  # This is very hard
        # order_name = ["gcc", "vqav2", "coco", "textvqa", "textcaps", "okvqa", "evjvqa"]  # This is less hard
    if order_lang is None:
        order_lang = ["en", "vi", "mixed"]

    def order_criteria(item: dict):
        return order_name.index(item["d_name"]), order_lang.index(item["d_lang"])

    dataset_list.sort(key=order_criteria)
    return dataset_list


class PadDataset(Dataset):
    def __init__(self, dataset: list):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def pad_dataset_list(
    dataset_list: list, num_devices=8, batch_size=8, accumulation_step=8
) -> list:
    """Add extra dataset so the total length of all dataset in the list
    is divided by num_devices * batch_size * accumulation_step


    Parameters
    ----------
    dataset_list : list
        the list of all datasets that is returned by `create_all_dataset_list()` or `order_dataset_list()`
    num_devices : int, optional
        obvious, by default 8
    batch_size : int, optional
        obvious, by default 8
    accumulation_step : int, optional
        obvious, by default 8

    Returns
    -------
    list
        List of dict. Each dict contain:
            - `d_name`: a valid dataset in toml file
            - `d_lang`: a valid iso639-1 code
            - `d_object`: a object of dataset class
    """
    total_length = 0
    for i in dataset_list:
        total_length += len(i["d_object"])

    effective_batch_size = num_devices * batch_size * accumulation_step
    to_add_length = effective_batch_size - total_length % effective_batch_size
    last_dataset = dataset_list[-1]
    assert to_add_length >= len(
        last_dataset
    ), "Last dataset must be longer than effective batch size"

    pad_dataset = PadDataset(
        [last_dataset["d_object"][i] for i in range(to_add_length)]
    )
    dataset_list.append(
        {
            "d_object": pad_dataset,
            "d_lang": last_dataset["d_lang"],
            "d_name": last_dataset["d_name"],
        }
    )
    return dataset_list
