from dataclasses import dataclass

from dataclass_wizard import JSONWizard
from dataclass_wizard.enums import LetterCase


@dataclass
class CCSBUAlign:
    root_dir: str


@dataclass
class COCO:
    image_root_dir: str
    metadata_root_dir: str


@dataclass
class EVJVQA:
    root_dir: str


@dataclass
class OKVQA:
    image_root_dir: str
    metadata_root_dir: str


@dataclass
class TextCaps:
    image_root_dir: str
    metadata_root_dir: str


@dataclass
class TextVQA:
    image_root_dir: str
    metadata_root_dir: str


@dataclass
class VQAv2:
    image_root_dir: str
    metadata_root_dir: str


@dataclass
class GCC:
    gcc_vi_dir: str
    llava_gcc_dir: str


@dataclass
class ConfigDataDir(JSONWizard):
    class _(JSONWizard.Meta):
        # Sets the target key transform to use for serialization;
        # defaults to `camelCase` if not specified.
        key_transform_with_load = LetterCase.SNAKE
        key_transform_with_dump = LetterCase.SNAKE

    cc_sbu_align: CCSBUAlign
    coco: COCO
    evjvqa: EVJVQA
    okvqa: OKVQA
    textcaps: TextCaps
    textvqa: TextVQA
    vqav2: VQAv2
    gcc: GCC
