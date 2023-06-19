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
        super().__init__()
