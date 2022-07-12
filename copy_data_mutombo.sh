#!/bin/bash


if [ ! -d "$BASE_PATH"/opensphere/data ]; then

  DATA_PATH_TRAIN="$BASE_PATH"/opensphere/data/train
  DATA_PATH_VAL="$BASE_PATH"/opensphere/data/val
  mkdir -p "$DATA_PATH_TRAIN"
  mkdir -p "$DATA_PATH_VAL"
  tb-cli get /prod/uface/mutombo_detections.tar "$BASE_PATH"/opensphere/data/mutombo_detections.tar
  tar -xvf "$BASE_PATH"/opensphere/data/mutombo_detections.tar
  mv "$BASE_PATH"/opensphere/data/mutombo_detections DATA_PATH_TRAIN
  python "$BASE_PATH"/opensphere/get_detected_mutombo_file_list.py

  cp /mnt/share/deeplearning_uface-face-detection/datasets/validation.tar "$DATA_PATH_VAL"
  tar xvf "$DATA_PATH_VAL"/validation.tar -C "$DATA_PATH_VAL"
  rm "$DATA_PATH_VAL"/validation.tar

fi