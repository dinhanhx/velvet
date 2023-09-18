import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig, BloomTokenizerFast
from transformers.models.convnext import ConvNextImageProcessor
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Config,
    ConvNextV2Model,
)

from velvet.model import VisualBloom


@dataclass
class ImageFeatureCollator:
    image_processor: ConvNextImageProcessor
    image_model: ConvNextV2Model

    def __call__(self, batch_image: List[Image.Image]):
        image_inputs = self.image_processor(batch_image, return_tensors="pt")

        with torch.no_grad():
            image_outputs = self.image_model(**image_inputs)
        image_features = image_outputs["last_hidden_state"]

        image_features = rearrange(image_features, "b c h w -> b h w c")
        image_features = rearrange(image_features, "b h w c -> b (h w) c")

        image_attentions = torch.ones(image_features.size()[:-1], dtype=torch.long)
        return image_features, image_attentions


if __name__ == "__main__":
    model_dir = Path("big_model_logs/lightning_logs/version_0")

    root_dir = Path("/mnt/storage/data/crossmodal-3600")
    image_dir = root_dir / "images"
    metadata_dir = root_dir / "captions.jsonl"

    metadata_df = pd.read_json(metadata_dir, lines=True)

    image_model_name = "facebook/convnextv2-large-22k-224"
    image_config = ConvNextV2Config.from_pretrained(image_model_name)
    image_processor = ConvNextImageProcessor.from_pretrained(image_model_name)
    image_model = ConvNextV2Model.from_pretrained(image_model_name)
    image_feature_collator = ImageFeatureCollator(image_processor, image_model)

    bloom_model_name = "bigscience/bloomz-1b7"
    bloom_config = BloomConfig.from_pretrained(bloom_model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(bloom_model_name)
    tokenizer.padding_side = "right"

    bert_config = BertConfig(
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        is_decoder=True,
        add_cross_attention=True,
    )

    visual_bloom = VisualBloom(
        image_config, bert_config, bloom_config, bloom_model_name
    )
    visual_bloom.load_state_dict(
        torch.load(str(model_dir.joinpath("checkpoints/visual_bloom.torch")))
    )
    visual_bloom.eval()

    language_list = ["vi", "en"]
    output_dict = {}
    for item in tqdm(metadata_df["image/key"].items()):
        index = item[0]
        image_file = image_dir / (item[1] + ".jpg")
        image = Image.open(image_file).convert("RGB")

        image_features, image_attentions = image_feature_collator([image])

        output_item = {}
        for language in language_list:
            instruction = f"Generate caption in {language}:"
            instruction_inputs = tokenizer([instruction], return_tensors="pt")

            language_output = visual_bloom.generate(
                image_features,
                image_attentions,
                instruction_inputs["input_ids"],
                instruction_inputs["attention_mask"],
            )

            raw_output = language_output[0]
            raw_output = tokenizer.decode(language_output[0], skip_special_tokens=True)
            cooked_output = raw_output.split(".")[0]

            output_item[language] = cooked_output
        output_dict[index] = output_item

    with open(
        str(model_dir.joinpath("crossmodal3600.json")),
        "w",
        encoding="utf-8",
    ) as jsonfile:
        json.dump(output_dict, jsonfile, ensure_ascii=False, indent=4)
