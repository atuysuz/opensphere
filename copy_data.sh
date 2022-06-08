#!/bin/bash


if [ ! -d "$BASE_PATH"/facerec_data ]; then

  DATA_PATH="$BASE_PATH"/opensphere/data
  mkdir "$DATA_PATH"
  cp /mnt/share/deeplearning_uface-face-detection/datasets/vggface2/vggface2.tar "$DATA_PATH"
  cp /mnt/share/deeplearning_uface-face-detection/datasets/lfw/lfw-deepfunneled.tgz "$DATA_PATH"
  tar xvf "$DATA_PATH"/vggface2.tar -C "$DATA_PATH"
  tar xvf "$DATA_PATH"/lfw-deepfunneled.tgz -C "$DATA_PATH"
  rm "$DATA_PATH"/vggface2.tar
  rm "$BASE_PATH"/facerec_data/lfw-deepfunneled.tgz
  cp /mnt/share/deeplearning_uface-face-detection/datasets/lfw/lfw_test_pair.txt "$BASE_PATH"/facerec_data
fi