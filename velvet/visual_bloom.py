import torch
from torch import nn
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig, BloomForCausalLM
from transformers.models.convnextv2 import ConvNextV2Config

from velvet.cutie import Cutie


class VisualBloom(nn.Module):
    """A BLOOM-based model that can take image inputs"""

    def __init__(
        self,
        convnextv2_config: ConvNextV2Config,
        bert_config: BertConfig,
        bloom_config: BloomConfig,
    ) -> None:
        super().__init__()

        # VisualBLoom doesn't have linear layers to project a model's input to other model's space.
        # Therefore, all models must return same embedding dimensions = 1024
        assert (
            convnextv2_config.hidden_sizes[-1]
            == bert_config.hidden_size
            == bloom_config.hidden_size
            == 1024
        ), "Something horrible just happen. All embedding dimensions must be 1024"

        self.cutie_model = Cutie(bert_config)

        # Load and freeze BLOOM model
        self.bloom_model = BloomForCausalLM.from_pretrained(bloom_config.name_or_path)
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
        cutie_output = self.cutie_model(
            image_features=image_features,
            image_attentions=image_attentions,
            instruction_embeds=instruction_embeds,
            instruction_attention_mask=instruction_attention_mask,
        )
        cutie_attentions = torch.ones(cutie_output.size()[:-1], dtype=torch.long)
        cutie_labes = torch.full(
            cutie_output.size()[:-1], -100, dtype=language_model_labels.dtype
        )

        language_model_embeds = self.bloom_model.transformer.word_embeddings(
            language_model_input_ids
        )

        cat_embeds = torch.cat([cutie_output, language_model_embeds], dim=1)
        cat_attentions = torch.cat(
            [cutie_attentions, language_model_attention_mask], dim=1
        )
        cat_labels = torch.cat([cutie_labes, language_model_labels], dim=1)

        bloom_outputs = self.bloom_model(
            inputs_embeds=cat_embeds, attention_mask=cat_attentions, labels=cat_labels
        )
        return bloom_outputs
