import warnings

import torch
from torch import nn
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig, BloomForCausalLM
from transformers.models.convnextv2 import ConvNextV2Config

from velvet.model.cutie import Cutie


class VisualBloom(nn.Module):
    """A BLOOM-based model that can take image inputs"""

    def __init__(
        self,
        convnextv2_config: ConvNextV2Config,
        bert_config: BertConfig,
        bloom_config: BloomConfig,
        bloom_name: str,
    ) -> None:
        super().__init__()

        if (
            convnextv2_config.hidden_sizes[-1]
            == bert_config.hidden_size
            == bloom_config.hidden_size
        ):
            self.use_projection = False
            warnings.warn(
                "All embedding dimensions are equal. No linear projection layers are created."
            )
        else:
            self.use_projection = True
            self.text_to_cutie = nn.Linear(
                bloom_config.hidden_size, bert_config.hidden_size
            )
            self.image_to_cutie = nn.Linear(
                convnextv2_config.hidden_sizes[-1], bert_config.hidden_size
            )
            self.cutie_to_text = nn.Linear(
                bert_config.hidden_size, bloom_config.hidden_size
            )

        self.cutie_model = Cutie(bert_config)

        # Load and freeze BLOOM model
        self.bloom_model = BloomForCausalLM.from_pretrained(bloom_name)
        for param in self.bloom_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        # Image model outputs - Q-former inputs
        image_features: torch.Tensor,
        image_attentions: torch.Tensor,
        # Q-former inputs
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        # Frozen language model inputs
        language_model_input_ids: torch.Tensor,
        language_model_attention_mask: torch.Tensor,
        language_model_labels: torch.Tensor,
    ):
        instruction_embeds = self.bloom_model.transformer.word_embeddings(
            instruction_input_ids
        )
        instruction_embeds = self.bloom_model.transformer.word_embeddings_layernorm(
            instruction_embeds
        )

        if self.use_projection:
            image_features = self.image_to_cutie(image_features)
            instruction_embeds = self.text_to_cutie(instruction_embeds)

        cutie_output = self.cutie_model(
            image_features=image_features,
            image_attentions=image_attentions,
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
        )

        if self.use_projection:
            cutie_output = self.cutie_to_text(cutie_output)

        cutie_attentions = self.cutie_model.query_attentions.expand(
            cutie_output.size(0), -1
        ).to(cutie_output.device)
        cutie_labels = self.cutie_model.query_labels.expand(
            cutie_output.size(0), -1
        ).to(cutie_output.device)

        language_model_embeds = self.bloom_model.transformer.word_embeddings(
            language_model_input_ids
        )
        language_model_embeds = self.bloom_model.transformer.word_embeddings_layernorm(
            language_model_embeds
        )

        cat_embeds = torch.cat([cutie_output, language_model_embeds], dim=1)
        cat_attentions = torch.cat(
            [cutie_attentions, language_model_attention_mask], dim=1
        )
        cat_labels = torch.cat([cutie_labels, language_model_labels], dim=1)

        bloom_outputs = self.bloom_model(
            inputs_embeds=cat_embeds, attention_mask=cat_attentions, labels=cat_labels
        )
        return bloom_outputs

    @torch.no_grad()
    def generate(
        self,
        # Image model outputs - Q-former inputs
        image_features: torch.Tensor,
        image_attentions: torch.Tensor,
        # Q-former inputs
        instruction_input_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
    ):
        instruction_embeds = self.bloom_model.transformer.word_embeddings(
            instruction_input_ids
        )
        instruction_embeds = self.bloom_model.transformer.word_embeddings_layernorm(
            instruction_embeds
        )

        if self.use_projection:
            image_features = self.image_to_cutie(image_features)
            cutie_instruction_embeds = self.text_to_cutie(instruction_embeds)

        cutie_output = self.cutie_model(
            image_features=image_features,
            image_attentions=image_attentions,
            instruction_embeds=cutie_instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
        )

        if self.use_projection:
            cutie_output = self.cutie_to_text(cutie_output)

        cutie_attentions = self.cutie_model.query_attentions.expand(
            cutie_output.size(0), -1
        ).to(cutie_output.device)

        cat_embeds = torch.cat([cutie_output, instruction_embeds], dim=1)
        cat_attentions = torch.cat(
            [cutie_attentions, instruction_attention_mask], dim=1
        )

        language_output = self.bloom_model.generate(
            inputs_embeds=cat_embeds,
            attention_mask=cat_attentions,
            max_length=105,
            penalty_alpha=0.6,
            top_k=4,
        )
        return language_output
