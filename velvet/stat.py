import math
from dataclasses import dataclass
from typing import Type, Union

from torch.utils.data import Dataset
from transformers.models.bloom import BloomTokenizerFast

from velvet.dataset import EVJVQA
from velvet.instruction_template import InstructionTemplate


@dataclass
class MeasureTokenLength:
    tokenizer: BloomTokenizerFast
    template_class: Type[InstructionTemplate]
    dataset: Union[EVJVQA, Dataset]

    def stat(self):
        len_dataset = len(self.dataset)
        instruction_len_list = []
        response_len_list = []
        for i in range(len_dataset):
            item = self.dataset[i]
            instruction, response = self.template_class.make_instruction_response_pair(
                question=item["question"],
                answer=item["answer"],
                language=item["language"],
            )
            instruction_len_list.append(len(self.tokenizer.encode(instruction)))
            response_len_list.append(len(self.tokenizer.encode(response)))

        make_stat_dict = lambda d: {  # noqa
            "min": min(d),
            "avg": math.ceil(sum(d) / len(d)),
            "max": max(d),
        }
        stat_dict = {
            "instruction_len": make_stat_dict(instruction_len_list),
            "response_len": make_stat_dict(response_len_list),
        }
        return stat_dict
