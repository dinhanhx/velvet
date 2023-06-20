from typing import Union

from torch.utils.data import Dataset

from .cc_sbu_align import CCSBUAlign
from .coco import COCO
from .evjvqa import EVJVQA
from .okvqa import OKVQA
from .textcaps import TextCaps
from .textvqa import TextVQA
from .vqav2 import VQAv2

UnionVelvetDataset = Union[
    Dataset, CCSBUAlign, COCO, EVJVQA, TextCaps, TextVQA, VQAv2, OKVQA
]
