#!/bin/bash
cd /home/darshan/MS/
conda init
conda activate venv

cd models/unetpp/scripts
mkdir -p /home/darshan/MS/models/unetpp/logs

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-5 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run1.log 2>&1


LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-5 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run2.log 2>&1


LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-4 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run3.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-4 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run4.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-3 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run5.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-3 --optimizer adam --skip_blank_ratio 0.975 --lr_schedule fixed --gpu 0 > /home/darshan/MS/models/unetpp/logs/run6.log 2>&1