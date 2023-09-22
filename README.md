Velvet
======

![Alt text](powered-by-wine-magic.svg)

> Red velvet cake is my all-time favorite cake. It’s perfect for any holiday or celebration, and it always looks so elegant!

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

# Introduction

This contains all the code (training, data loading) for the prompting vision language model discussed in the master thesis "Multitask Multilingual Vision Language: Image and Text Generation".

To increase the language range with Vietnamese as our main focus, we have translated 6 common English datasets for image captioning and visual question answering into Vietnamese. The 6 datasets are CC3M, COCO, VQAv2, OK-VQA, TextCaps, TextVQA. By using Google Translate, these datasets collectively contain millions of image-text pairs in each language (English and Vietnamese). These datasets are available at [huggingface.co/dinhanhx](https://huggingface.co/dinhanhx).

We have proposed a prompting vision language model (**this repository**) which is trained on these datasets. The model can caption images and answer questions related to images. The model is modular and has 3 components: a frozen image model (ConvNeXt V2), a frozen text model (BLOOMZ), and a Cutie model. Cutie model is meant to query useful information from visual features (by image model) for the text model.

# Citation

If you use this source code or model weights or theory, please cite it as below.
```
@software{dinhanhx_Velvet_2023,
	title        = {{Velvet}},
	author       = {dinhanhx},
	year         = 2023,
	month        = sep,
	url          = {https://github.com/dinhanhx/velvet},
	license      = {MIT}
}
```

# Results

It's unfortunately not good. Our largest model achieves CIDEr of Crossmodal-3600 of 0.435 in English and 0.318 in Vietnamese. It also gets 0.3404 F1 and 0.1135 BLEU on the private test of EVJVQA.

# Project setup

## Dependencies setup

See [this pip requirements](https://gist.github.com/dinhanhx/2cf2c8b3dbf45db8d722bca5c098d3dd#pip-requirements) to setup on TPU v3-8 or v4-8.

## Data setup

⚠ Please take a look at this file `configs/data_dir.toml` first

Given that the folder to download data is `/mnt/storage/data/`, in this folder, run the following commands,
```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/datasets/dinhanhx/evjvqa
git clone https://huggingface.co/datasets/dinhanhx/TextVQA-vi
git clone https://huggingface.co/datasets/dinhanhx/TextCaps-vi
git clone https://huggingface.co/datasets/dinhanhx/OK-VQA-multilang
git clone https://huggingface.co/datasets/dinhanhx/VQAv2-vi
git clone https://huggingface.co/datasets/dinhanhx/cc_sbu_align_multilang
git clone https://huggingface.co/datasets/dinhanhx/coco-2017-vi
git clone https://huggingface.co/datasets/dinhanhx/gcc-vi
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
```

Now setup `LLaVA-CC3M-Pretrain-595K`,
```sh
cd LLaVA-CC3M-Pretrain-595K/
mkdir images
unzip -q images.zip -d images
```

Now setup `coco-2017-images`,
```sh
mkdir coco-2017-images
cd coco-2017-images
curl -JO http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
```

Now setup `OpenImages`,
```sh
mkdir OpenImages
cd OpenImages
curl -JO https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
unzip -q train_val_images.zip
```

Now run some python scripts to make data usable by velvet, 
don't forget do back to root of this repository,
```sh
python scripts/<do_something>.py
```
where `<do_something>.py` is a file in `scripts/`

⚠ Please read the files (including `scripts/readme.md`) before running, especially things related to paths.

## Experiment train

⚠ Our training code can be used on GPU too. It's not TPU exclusive. Please read `pl.Trainer` in `velvet_exp/run_all.py` carefully before running on any hardware.

### Pre run

for TPU v3-8
```sh
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

for TPU v4-8
```sh
export PJRT_DEVICE=TPU
```

### Main run

```sh
# Replace <name> by config name
python -m velvet_exp.run_all configs/experiments/<name>.json
```

```sh
# Kill leftover processes
pgrep -f "python -m velvet_exp.run_all" | xargs kill -9
```

## PyTorch Lightning CKPT file extraction

PyTorch Lightning (after and during training) produces a ckpt file (which can be opened by Torch). It contains everything **including** the model weight. To extract the model weight only, please refer the following code (which is similar to `velvet_exp/run_all.py`)

<details>
    <summary>Click me</summary>

```python
import json

import torch
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2Config

from velvet_exp.run_all import Wrapper

experiment_config = json.load(open("configs/experiments/big_model.json", "r"))

image_model_name = experiment_config.get("image_model_name", None)
if image_model_name is None:
    image_model_name = "facebook/convnextv2-base-22k-224"

image_config = ConvNextV2Config.from_pretrained(image_model_name)

bloom_model_name = experiment_config.get("bloom_model_name", None)
if bloom_model_name is None:
    bloom_model_name = "bigscience/bloomz-560m"

bloom_config = BloomConfig.from_pretrained(bloom_model_name)

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
    warmup_steps=experiment_config.get("warmup_steps", None),
)

wrapper.load_state_dict(torch.load("big_model_logs/lightning_logs/version_0/checkpoints/last.ckpt")["state_dict"])

torch.save(wrapper.visual_bloom.state_dict(), "big_model_logs/lightning_logs/version_0/checkpoints/visual_bloom.torch")
```

</details>

## Demo

⚠ Please make sure that you have done the extraction or you have obtained [the model weight here](https://drive.google.com/file/d/1g3c9INmUyYCnYbrBTmnjrGEc56iuQ_hz/view?usp=sharing)

The following code doesn't use GPU or TPU. To change the image, please look at variable `url`. To change the prompt, please read the last for loop.

<details>
    <summary>Click me</summary>

```python
import requests
import torch
from PIL import Image
from transformers.models.bert import BertConfig
from transformers.models.bloom import BloomConfig, BloomTokenizerFast
from transformers.models.convnext import ConvNextImageProcessor
from transformers.models.convnextv2.modeling_convnextv2 import (
    ConvNextV2Config,
    ConvNextV2Model,
)

from velvet.collator import ImageFeatureCollator
from velvet.model import VisualBloom

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
    image_config, bert_config, bloom_config, bloom_model_name, use_frozen_bloom=False
)
visual_bloom.load_state_dict(
    torch.load("big_model_logs/lightning_logs/version_0/checkpoints/visual_bloom.torch")
)
visual_bloom = visual_bloom.eval()

url = "https://i.imgur.com/Y2vIAJp.jpg"

language_list = ["en", "vi"]
for language in language_list:
    # instruction = (
    #     f"Generate answer in {language}: what is the color of the sky?"
    #     if language == "en"
    #     else f"Generate answer in {language}: màu sắc của bầu trời là gì?"
    # )
    instruction = f"Generate caption in {language}:"
    print(instruction)
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    image_features, image_attentions = image_feature_collator([image])
    instruction_inputs = tokenizer([instruction], return_tensors="pt")

    language_output = visual_bloom.generate(
        image_features,
        image_attentions,
        instruction_inputs["input_ids"],
        instruction_inputs["attention_mask"],
    )

    human_output = tokenizer.decode(language_output[0], skip_special_tokens=True)
    print(human_output.split(".")[0])
```

</details>