import torch
from torch import nn
from transformers.models.bert import BertConfig, BertModel


class IdentityForBertEmbeddings(nn.Module):
    """To skip all BertEmbeddings because another text embeddings provided by another model are used
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, **bert_embeddings_args):
        inputs_embeds = bert_embeddings_args.get("inputs_embeds", None)
        return inputs_embeds


class Cutie(nn.Module):
    """Cutie - Qt - Query Transformer - Q-Former

    Cutie is motivated by the underlying theoretical foundations of Q-Former presented in BLIP-2 https://arxiv.org/abs/2301.12597
    It should be noted that Cutie differs from the specific approach described in the aforementioned paper
    Both Cutie and Q-former have Query tokens.
    Cutie uses the same unmodified BERT.
    Q-former modifies BERT to behave differently on some tasks.
    """

    def __init__(
        self,
        bert_config: BertConfig,
        max_query_length: int = 32,
        language_model_ignore_label: int = -100,
    ) -> None:
        assert bert_config.is_decoder, "BERT must be a decoder"
        assert bert_config.add_cross_attention, "BERT must have cross attention layer"
        super().__init__()
        self.bert_model = BertModel(bert_config, add_pooling_layer=False)
        self.bert_model.embeddings = IdentityForBertEmbeddings()

        self.query_tokens = nn.Parameter(
            torch.zeros(1, max_query_length, bert_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=bert_config.initializer_range)
        self.query_attentions = torch.ones(
            self.query_tokens.size()[:-1], dtype=torch.long
        )
        self.query_labels = torch.full(
            self.query_tokens.size()[:-1], language_model_ignore_label, dtype=torch.long
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_attentions: torch.Tensor,
        instruction_embeds: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
    ):
        batch_size = image_features.size(0)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1).to(
            self.query_tokens.device
        )
        query_attentions = self.query_attentions.expand(batch_size, -1).to(
            self.query_tokens.device
        )

        cat_embeds = torch.cat([query_tokens, instruction_embeds], dim=1)
        cat_attentions = torch.cat(
            [query_attentions, instruction_attention_mask], dim=1
        )

        bert_outputs = self.bert_model(
            inputs_embeds=cat_embeds,
            attention_mask=cat_attentions,
            encoder_hidden_states=image_features,
            encoder_attention_mask=image_attentions,
        )
        cutie_output = bert_outputs.last_hidden_state[:, : query_tokens.size(1), :]
        return cutie_output
