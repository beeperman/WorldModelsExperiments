#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=""

for i in `seq 1 40`;
do
  echo worker $i
  python train_extract.py -s 5 -t 50 &
  sleep 1.0
done
