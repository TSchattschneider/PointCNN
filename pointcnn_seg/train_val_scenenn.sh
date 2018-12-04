#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64 python3 ../train_val_scenenn.py
