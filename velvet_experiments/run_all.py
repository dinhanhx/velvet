import json
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig, BloomTokenizerFast
from transformers.models.convnext import ConvNextImageProcessor
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Config,
    ConvNextV2Model,
)
from transformers.optimization import get_cosine_schedule_with_warmup

from velvet.collator import ImageTextCollator
from velvet.dataset import (
    create_all_dataset_list,
    filter_dataset_list,
    order_dataset_list,
    pad_dataset_list,
)
from velvet.model import VisualBloom


class Wrapper(pl.LightningModule):
    def __init__(
        self,
        image_config: ConvNextV2Config,
        bert_config: BertConfig,
        bloom_config: BloomConfig,
        bloom_name: str = "bigscience/bloomz-560m",
        learning_rate=5e-5,
        warmup_ratio=0.2,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

        self.visual_bloom = VisualBloom(
            image_config, bert_config, bloom_config, bloom_name
        )

    def training_step(self, batch) -> STEP_OUTPUT:
        loss = self.visual_bloom(**batch).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        opt = AdamW(self.parameters(), self.learning_rate)
        lrs = get_cosine_schedule_with_warmup(
            opt,
            self.trainer.estimated_stepping_batches * self.warmup_ratio,
            self.trainer.estimated_stepping_batches,
        )
        return [opt], [lrs]


if __name__ == "__main__":
    experiment_config = json.load(
        open("configs/experiments/easier_dataset_order.json", "r")
    )

    global_seed = 1312
    pl.seed_everything(global_seed)

    dataset_list = create_all_dataset_list("configs/data_dir.toml", global_seed)
    dataset_list = order_dataset_list(
        dataset_list,
        experiment_config["dataset_config"]["order_name"],
        experiment_config["dataset_config"]["order_lang"],
    )
    if len(experiment_config["dataset_config"]["ignore_name"]) != 0:
        dataset_list = filter_dataset_list(
            dataset_list, experiment_config["dataset_config"]["ignore_name"]
        )
    if experiment_config["dataset_config"]["do_merge_lang"]:
        pass
    dataset_list = pad_dataset_list(
        dataset_list,
        experiment_config["num_devices"],
        experiment_config["batch_size"],
        experiment_config["accumulation_step"],
    )

    dataset = ConcatDataset([i["d_object"] for i in dataset_list])

    image_config = ConvNextV2Config.from_pretrained("facebook/convnextv2-base-22k-224")
    image_processor = ConvNextImageProcessor.from_pretrained(
        "facebook/convnextv2-base-22k-224"
    )
    image_model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")

    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloomz-560m")
    bloom_config = BloomConfig.from_pretrained("bigscience/bloomz-560m")

    collator = ImageTextCollator(
        image_processor=image_processor,
        image_model=image_model,
        tokenizer=tokenizer,
        max_instruction_len=61,
        max_instruction_response_len=105,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=experiment_config["batch_size"],
        collate_fn=collator,
        num_workers=24,
        prefetch_factor=4,
    )

    bert_config = BertConfig(
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        is_decoder=True,
        add_cross_attention=True,
    )

    wrapper = Wrapper(
        image_config=image_config,
        bert_config=bert_config,
        bloom_config=bloom_config,
        learning_rate=experiment_config["learning_rate"],
        warmup_ratio=experiment_config["warmup_ratio"],
    )

    trainer = pl.Trainer(
        enable_checkpointing=True,
        default_root_dir="experiment_logs",
        accelerator=experiment_config["hardware"]["type"],
        devices=experiment_config["num_devices"],
        precision="16-mixed",
        logger=[
            TensorBoardLogger("experiment_logs"),
        ],
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                every_n_train_steps=experiment_config["do_every_n_steps"],
                save_last=True,
            ),
        ],
        max_epochs=experiment_config["max_epochs"],
        accumulate_grad_batches=experiment_config["accumulation_step"],
        # strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=experiment_config["do_every_n_steps"],
        enable_model_summary=False,
    )

    trainer.fit(wrapper, dataloader)
