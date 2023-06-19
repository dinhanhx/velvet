from typing import Union

from torch.utils.data import Dataset

from .cc_sbu_align import CCSBUAlign
from .coco import COCO
from .evjvqa import EVJVQA
from .textcaps import TextCaps

UnionVelvetDataset = Union[Dataset, CCSBUAlign, COCO, EVJVQA, TextCaps]
