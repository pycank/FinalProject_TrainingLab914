#!/bin/bash

working_dir=.

while [[ $# -gt 0 ]]; do
  case $1 in
    -w|--working_dir)
      working_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -a|--audio_dir)
      audio_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out_dir)
      out_dir="$2"
      shift # past argument
      shift # past value
      ;;
#    --default)
#      DEFAULT=YES
#      shift # past argument
#      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z ${audio_dir+x} ];
then audio_dir=${working_dir}/slurp
fi
if [ -z ${out_dir+x} ];
then out_dir=${working_dir}/out
fi

echo "Prepare data:..."
echo "Augments: "
for i in working_dir audio_dir out_dir;
do
  echo "$i = ${!i}";
done
echo

echo "cd $working_dir"
cd "$working_dir" || exit
for i in working_dir audio_dir out_dir;
do
  echo "Make dir if not exists: ${!i}"
  mkdir -p ${!i};
done
echo

echo "downloading audio from kaggle..."
kaggle datasets download -d mrhakk/slurp-audio

echo "unzip..."
unzip -o -q  "$working_dir"/slurp-audio.zip -d "$working_dir"

echo "moving audio..."
mkdir -p "$working_dir"/slurp/slurp_synth
find "$working_dir"/slurp_synth/slurp_synth/ -type f -print0 | xargs -0 mv -t "$working_dir"/slurp/slurp_synth
mkdir -p "$working_dir"/slurp/slurp_real
find "$working_dir"/slurp_real/slurp_real/ -type f -print0 | xargs -0 mv -t "$working_dir"/slurp/slurp_real

echo "remove tmp..."
rm -rf "$working_dir"/slurp_synth
rm -rf "$working_dir"/slurp_real
echo

echo "download csv from kaggle..."
kaggle datasets download -d mrhakk/slurp-lb
unzip "$working_dir"/slurp-lb.zip -d "$working_dir"/out

echo "rename it..."
for f in "$out_dir"/*.csv; do mv "$f" "$(echo "$f" | sed s/typedirect/type=direct/)"; done
for f in "$out_dir"/*.csv; do mv "$f" "$(echo "$f" | sed s/sampling0/sampling=0/)"; done
for f in "$out_dir"/*.csv; do mv "$f" "$(echo "$f" | sed s/sample0/sample=0/)"; done
