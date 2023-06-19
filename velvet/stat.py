import math
from dataclasses import dataclass, field

from dataclass_wizard import JSONWizard
from dataclass_wizard.enums import LetterCase
from transformers.models.bloom import BloomTokenizerFast

from velvet.dataset import UnionVelvetDataset


@dataclass
class MeasureTokenLength(JSONWizard):
    class _(JSONWizard.Meta):
        # Sets the target key transform to use for serialization;
        # defaults to `camelCase` if not specified.
        key_transform_with_load = LetterCase.SNAKE
        key_transform_with_dump = LetterCase.SNAKE

    instruction_len_stat_dict: dict = field(default_factory=dict)
    response_len_stat_dict: dict = field(default_factory=dict)
    tokenizer_name: str = ""
    dataset_class_name: str = ""

    def make_stat(self, tokenizer: BloomTokenizerFast, dataset: UnionVelvetDataset):
        len_dataset = len(dataset)
        instruction_len_list = []
        response_len_list = []
        for i in range(len_dataset):
            item = dataset[i]
            instruction_len_list.append(len(tokenizer.encode(item["_instruction_"])))
            response_len_list.append(len(tokenizer.encode(item["_response_"])))

        make_stat_dict = lambda d: {  # noqa
            "min": min(d),
            "avg": math.ceil(sum(d) / len(d)),
            "max": max(d),
        }
        self.instruction_len_stat_dict = make_stat_dict(instruction_len_list)
        self.response_len_stat_dict = make_stat_dict(response_len_list)

        self.tokenizer_name = tokenizer.name_or_path
        self.dataset_class_name = dataset.__class__.__name__
