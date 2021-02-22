#! /usr/bin/bash env

num_gpu=($(nvidia-smi -L | wc -l))
num_worker=${num_gpu}

cd ../
python build_rawframes.py ../data/customdata/videos/ ../data/customdata/rawframes/ --level 2  --ext mp4 --num_gpu ${num_gpu} --num_worker ${num_worker}
echo "Raw frames (RGB only) generated for train and val set"

cd hmdb51/