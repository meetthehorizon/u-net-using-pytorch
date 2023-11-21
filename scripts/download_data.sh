#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade

if [[ ! -f lung-mask-image-dataset.zip ]]; then
  kaggle datasets download -d newra008/lung-mask-image-dataset
fi

if [ -z "$1" ]; then
  echo "Error: please specify the path to the data directory"
  exit 1
fi

TARGET_DIR=$1

if [[ ! -d "$TARGET_DIR" ]]; then
  mkdir -p "$TARGET_DIR"
  unzip lung-mask-image-dataset.zip -d "$TARGET_DIR"
  mv "$TARGET_DIR/ChestXray"/* "$TARGET_DIR"
  rm -rf "$TARGET_DIR/ChestXray"
else
  echo "Error: Target directory '$TARGET_DIR' not empty. Have you already downloaded the data?"  
fi

