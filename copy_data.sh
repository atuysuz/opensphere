#!/bin/bash


if [ ! -d "$BASE_PATH"/opensphere/data ]; then

  DATA_PATH_TRAIN="$BASE_PATH"/opensphere/data/train
  DATA_PATH_VAL="$BASE_PATH"/opensphere/data/val
  mkdir -p "$DATA_PATH_TRAIN"
  mkdir -p "$DATA_PATH_VAL"
  cp /mnt/share/deeplearning_uface-face-detection/datasets/vggface2/vggface2.tar "$DATA_PATH_TRAIN"
  cp /mnt/share/deeplearning_uface-face-detection/datasets/validation.tar "$DATA_PATH_VAL"
  tar xvf "$DATA_PATH_TRAIN"/vggface2.tar -C "$DATA_PATH_TRAIN"
  tar xvf "$DATA_PATH_VAL"/validation.tar -C "$DATA_PATH_VAL"
  rm "$DATA_PATH_TRAIN"/vggface2.tar
  rm "$DATA_PATH_VAL"/validation.tar
fi