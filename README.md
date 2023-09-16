Velvet
======

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