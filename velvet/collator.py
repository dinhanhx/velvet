from dataclasses import dataclass
from typing import Any, List

import torch
from einops import rearrange
from PIL import Image
from transformers.models.bloom import BloomTokenizerFast
from transformers.models.convnext import ConvNextImageProcessor
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Model
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class ImageFeatureCollator:
    image_processor: ConvNextImageProcessor
    image_model: ConvNextV2Model

    def __call__(self, batch_image: List[Image.Image]):
        return self.tensorize_batch_image(batch_image=batch_image)

    def tensorize_batch_image(self, batch_image: List[Image.Image]):
        image_inputs = self.image_processor(batch_image, return_tensors="pt")

        with torch.no_grad():
            image_outputs = self.image_model(**image_inputs)
        image_features = image_outputs["last_hidden_state"]

        image_features = rearrange(image_features, "b c h w -> b h w c")
        image_features = rearrange(image_features, "b h w c -> b (h w) c")

        image_attentions = torch.ones(image_features.size()[:-1], dtype=torch.long)
        return image_features, image_attentions


@dataclass
class TextCollator:
    tokenizer: BloomTokenizerFast
    max_instruction_len: int
    max_instruction_response_len: int

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def tensorize_batch_text(
        self, batch_instruction: List[str], batch_response: List[str]
    ):
        # This function is inspired by InstructBLIP
        # However this function is more well-written
        # https://github.com/salesforce/LAVIS/blob/59273f651b9bffb193d1b12a235e909e9f826dda/lavis/models/blip2_models/blip2_vicuna_instruct.py#L115-L238
        self.tokenizer.padding_side = "right"
        instruction_inputs = self.tokenizer(
            batch_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_instruction_len,
        )
        response_inputs = self.tokenizer(batch_response)

        assert (
            len(batch_instruction)
            == len(batch_response)
            == instruction_inputs["input_ids"].size(0)
            == len(response_inputs["input_ids"])
        ), "Something horrible just happen. All text-related batches have different size!"
        batch_size = len(batch_instruction)

        # Prepare inputs for Frozen language transformer decoder
        language_model_input_ids = []
        language_model_attention_mask = []
        instruction_length_list = []
        for i in range(batch_size):
            # To retrieve back length of instruction before being padded
            # attention_mask use 1 for token that is not padded
            # total of 1s == number tokens that are not pads - original instruction tokens
            instruction_length = instruction_inputs["attention_mask"][i].sum()
            instruction_length_list.append(instruction_length)

            language_model_input_ids.append(
                torch.cat(
                    [
                        instruction_inputs["input_ids"][i][:instruction_length],
                        torch.tensor(response_inputs["input_ids"][i], dtype=torch.long),
                        instruction_inputs["input_ids"][i][instruction_length:],
                    ]
                )
            )

            language_model_attention_mask.append(
                torch.cat(
                    [
                        instruction_inputs["attention_mask"][i][:instruction_length],
                        torch.tensor(
                            response_inputs["attention_mask"][i], dtype=torch.long
                        ),
                        instruction_inputs["attention_mask"][i][instruction_length:],
                    ]
                )
            )

        # Pad if input_length < self.max_instruction_response_len else truncate
        for i in range(batch_size):
            input_length = language_model_input_ids[i].size(0)
            if input_length == self.max_instruction_response_len:
                continue
            if input_length < self.max_instruction_response_len:
                to_add_length = self.max_instruction_response_len - input_length
                language_model_input_ids[i] = torch.cat(
                    [
                        language_model_input_ids[i],
                        torch.full(
                            (to_add_length,),
                            self.tokenizer.pad_token_id,
                            dtype=torch.long,
                        ),
                    ]
                )
                language_model_attention_mask[i] = torch.cat(
                    [
                        language_model_attention_mask[i],
                        torch.full((to_add_length,), 0, dtype=torch.long),
                    ]
                )
            if input_length > self.max_instruction_response_len:
                language_model_input_ids[i] = language_model_input_ids[i][
                    : self.max_instruction_response_len
                ]
                language_model_attention_mask[i] = language_model_attention_mask[i][
                    : self.max_instruction_response_len
                ]

        language_model_input_ids = torch.stack(language_model_input_ids)
        language_model_attention_mask = torch.stack(language_model_attention_mask)

        # Prepare labels for Frozen language transformer decoder
        # Every other token than response token is labeled as -100
        language_model_labels = language_model_input_ids.masked_fill(
            language_model_input_ids == self.tokenizer.pad_token_id, -100
        )
        for i in range(batch_size):
            language_model_labels[i][: instruction_length_list[i]] = -100

        return (
            # Inputs for Q-former
            instruction_inputs["input_ids"],
            instruction_inputs["attention_mask"],
            # Inputs for Frozen language transformer decoder
            language_model_input_ids,
            language_model_attention_mask,
            language_model_labels,
        )


@dataclass
class ImageTextCollator(ImageFeatureCollator, TextCollator):
    def __call__(self, batch_inputs) -> BatchEncoding:
        batch_image = []
        batch_instruction = []
        batch_response = []

        for item in batch_inputs:
            batch_image.append(Image.open(item["image_file"]).convert('RGB'))
            batch_instruction.append(item["_instruction_"])
            batch_response.append(item["_response_"])

        image_features, image_attentions = self.tensorize_batch_image(batch_image)
        (
            instruction_input_ids,
            instruction_attention_mask,
            language_model_input_ids,
            language_model_attention_mask,
            language_model_labels,
        ) = self.tensorize_batch_text(batch_instruction, batch_response)
        return BatchEncoding(
            {
                # Image model outputs - Q-former inputs
                "image_features": image_features,
                "image_attentions": image_attentions,
                # Q-former inputs
                "instruction_input_ids": instruction_input_ids,
                "instruction_attention_mask": instruction_attention_mask,
                # Frozen language model inputs
                "language_model_input_ids": language_model_input_ids,
                "language_model_attention_mask": language_model_attention_mask,
                "language_model_labels": language_model_labels,
            }
        )
