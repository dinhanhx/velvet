# Velvet

> Red velvet cake is my all-time favorite cake. It’s perfect for any holiday or celebration, and it always looks so elegant!

## Experiments

See [this pip requirements](https://gist.github.com/dinhanhx/2cf2c8b3dbf45db8d722bca5c098d3dd#pip-requirements) to setup on TPU v3-8, v4-8.

`export XRT_TPU_CONFIG="localservice;0;localhost:51011"` for TPU v3-8

`export PJRT_DEVICE=TPU` for TPU v4-8

`python -m velvet_exp.run_all configs/experiments/<name>.json`

`pgrep -f "python -m velvet_exp.run_all" | xargs kill -9`