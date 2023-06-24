#!/bin/bash

# Input directory (target to extract file from google drive folder)
out_dir="$1"

if [ $1 ]
then
  # Create the directory if it doesn't exist
  mkdir -p "$out_dir"

  # download csv files from driver
  gdown --folder 1Ntr_iD89rHPW6CGMVimGEBTRYOK5jw8z
  mv v1-ds-full/* $out_dir
  rm -rf v1-ds-full
  gdown --folder 10qssW9bxqht2qBysr5UUzyILIePm2dco
  mv v1-ds-sample-0.2-by-intent-only/* $out_dir
  rm -rf v1-ds-sample-0.2-by-intent-only

else
  echo "Not found out_dir"
fi
