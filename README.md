Velvet
======

![Alt text](powered-by-wine-magic.svg)

> Red velvet cake is my all-time favorite cake. It’s perfect for any holiday or celebration, and it always looks so elegant!

Research supported with Cloud TPUs from Google's TPU Research Cloud (TRC)

# Introduction

This contains all the code (training, data loading) for the prompting vision language model discussed in the master thesis "Multitask Multilingual Vision Language: Image and Text Generation"

# Results

It's unfortunately not good.

# Project setup

## Dependencies setup

See [this pip requirements](https://gist.github.com/dinhanhx/2cf2c8b3dbf45db8d722bca5c098d3dd#pip-requirements) to setup on TPU v3-8 or v4-8.

## Data setup

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

## Experiment run

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