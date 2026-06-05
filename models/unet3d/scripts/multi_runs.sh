#!/bin/bash
cd /home/darshan/MS/
conda init
conda activate venv

cd models/unet3d/scripts
mkdir -p /home/darshan/MS/models/unet3d/logs

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-5 --optimizer adam --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run1.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-4 --optimizer adam --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run2.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-3 --optimizer adam --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run3.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-5 --optimizer sgd --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run4.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-4 --optimizer sgd --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run5.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 1 --lr 1e-3 --optimizer sgd --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet3d/logs/run6.log 2>&1
