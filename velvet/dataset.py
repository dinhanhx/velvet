import json
import random
from pathlib import Path

from torch.utils.data import Dataset


class EVJVQA(Dataset):
    def __init__(self, root_dir: Path, shuffle_seed: int = None) -> None:
        """
        Parameters
        ----------
        root_dir : Path
            Directory where contain EVJVQA train-images/, lang_evjvqa_train.json
        shuffle_seed : int | None
            if shuffle_seed is None, don't shuffle dataset else shuffle according to seed value
        """
        self.root_dir = root_dir
        self.img_dir = root_dir.joinpath("train-images")
        self.json_file = root_dir.joinpath("lang_evjvqa_train.json")

        self.dataset = json.load(open(self.json_file))["annotations"]
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.dataset)

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
        return data
