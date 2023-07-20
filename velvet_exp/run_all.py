import json
from typing import Any, Union

import click
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
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
    create_validation_list,
    filter_dataset_list,
    order_dataset_list,
    pad_dataset_list,
)
from velvet.model import VisualBloom


class Wrapper(pl.LightningModule):
    def __init__(
        self,
        experiment_config: dict,
        image_config: ConvNextV2Config,
        bert_config: BertConfig,
        bloom_config: BloomConfig,
        bloom_name: str = "bigscience/bloomz-560m",
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.2,
        use_lrs: bool = True,
        warmup_steps: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.experiment_config = experiment_config
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.use_lrs = use_lrs
        self.warmup_steps = warmup_steps
        self.save_hyperparameters("experiment_config")

        self.visual_bloom = VisualBloom(
            image_config, bert_config, bloom_config, bloom_name
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.visual_bloom(**batch).loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Union[STEP_OUTPUT, None]:
        loss = self.visual_bloom(**batch).loss
        self.log("val_loss", loss)

    def configure_optimizers(self) -> Any:
        opt = AdamW(self.parameters(), self.learning_rate, weight_decay=0.05)
        opt_list = [opt]

        if self.use_lrs:
            calculated_warmup_steps = (
                self.trainer.estimated_stepping_batches * self.warmup_ratio
            )
            lrs = {
                # See this docs: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#:~:text=The%20lr_scheduler_config%20is%20a%20dictionary%20which%20contains%20the%20scheduler%20and%20its%20associated%20configuration.%20The%20default%20configuration%20is%20shown%20below.
                "scheduler": get_cosine_schedule_with_warmup(
                    opt,
                    self.warmup_steps if self.warmup_steps is not None else calculated_warmup_steps,  # type: ignore
                    self.trainer.estimated_stepping_batches,  # type: ignore
                ),
                "interval": "step",
                "frequency": 1,
            }
            lrs_list = [lrs]
        else:
            lrs_list = []

        return opt_list, lrs_list


@click.command()
@click.argument("experiment_config_file")
def main(experiment_config_file: str):
    experiment_config = json.load(open(experiment_config_file, "r"))

    global_seed = 1312
    pl.seed_everything(global_seed)

    dataset_list = create_all_dataset_list("configs/data_dir.toml", global_seed)
    if len(experiment_config["dataset_config"]["ignore_name"]) != 0:
        dataset_list = filter_dataset_list(
            dataset_list, experiment_config["dataset_config"]["ignore_name"]
        )
    dataset_list = order_dataset_list(
        dataset_list,
        experiment_config["dataset_config"]["order_name"],
        experiment_config["dataset_config"]["order_lang"],
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

    validation_list = create_validation_list("configs/data_dir.toml", global_seed)
    validation_dataset = ConcatDataset([i["d_object"] for i in validation_list])

    image_model_name = "facebook/convnextv2-base-22k-224"
    image_config = ConvNextV2Config.from_pretrained(image_model_name)
    image_processor = ConvNextImageProcessor.from_pretrained(image_model_name)
    image_model = ConvNextV2Model.from_pretrained(image_model_name)

    bloom_model_name = "bigscience/bloomz-560m"
    tokenizer = BloomTokenizerFast.from_pretrained(bloom_model_name)
    bloom_config = BloomConfig.from_pretrained(bloom_model_name)

    collator = ImageTextCollator(
        image_processor=image_processor,  # type: ignore
        image_model=image_model,  # type: ignore
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

    validation_dataloader = DataLoader(
        validation_dataset,
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
        experiment_config=experiment_config,
        image_config=image_config,  # type: ignore
        bert_config=bert_config,
        bloom_config=bloom_config,  # type: ignore
        bloom_name=bloom_model_name,
        learning_rate=experiment_config["learning_rate"],
        warmup_ratio=experiment_config["warmup_ratio"],
        use_lrs=experiment_config["use_learning_rate_scheduler"],
    )

    experiment_name = experiment_config["experiment_name"]
    save_root_dir = f"{experiment_name}_logs"
    trainer = pl.Trainer(
        enable_checkpointing=True,
        default_root_dir=save_root_dir,
        accelerator=experiment_config["hardware"]["type"],
        devices=experiment_config["num_devices"],
        precision="16-mixed",
        logger=[
            TensorBoardLogger(save_root_dir),
        ],
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(
                every_n_train_steps=experiment_config["do_every_n_steps"],
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        max_epochs=experiment_config["max_epochs"],
        accumulate_grad_batches=experiment_config["accumulation_step"],
        # strategy="ddp_find_unused_parameters_true",
        log_every_n_steps=experiment_config["do_every_n_steps"],
        enable_model_summary=False,
        val_check_interval=0.1,
    )

    ckpt_path: Union[str, None] = None
    trainer.fit(wrapper, dataloader, validation_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
