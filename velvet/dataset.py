import json
import random
from pathlib import Path
from typing import Type, Union

from torch.utils.data import Dataset

from velvet.instruction_template import (
    InstructionTemplate,
    ReverseVisualQuestionAnswerTemplate,
    VisualQuestionAnswerTemplate,
)


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

        self.dataset = json.load(open(self.json_file))["annotations"]
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
