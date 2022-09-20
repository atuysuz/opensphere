#!/bin/bash


if [ ! -d "$BASE_PATH"/opensphere/data ]; then

  DATA_PATH_VAL="$BASE_PATH"/opensphere/data/val
  mkdir -p "$DATA_PATH_VAL"
  cp /mnt/share/deeplearning_uface-face-detection/datasets/validation.tar "$DATA_PATH_VAL"
  tar xvf "$DATA_PATH_VAL"/validation.tar -C "$DATA_PATH_VAL"
  rm "$DATA_PATH_VAL"/validation.tar

fi