cd home/darshan/MS/
conda init
conda activate venv

cd models/scripts

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-5 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run1.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-4 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run2.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-3 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run3.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-5 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run4.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-4 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run5.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-3 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run6.log 2>&1


LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-5 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run7.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-4 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run8.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-3 --optimizer adam --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run9.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-5 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run10.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-4 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run11.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 4 --lr 1e-3 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run12.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-5 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run13.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-4 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run14.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 8 --lr 1e-3 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run15.log 2>&1


LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-5 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run16.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-4 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run17.log 2>&1

LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib /home/darshan/miniconda3/envs/venv/bin/python train.py --epochs 100 --batch_size 16 --lr 1e-3 --optimizer sgd --skip_blank_ratio 0.95 --lr_schedule fixed --gpu 1 > /home/darshan/MS/models/unet/logs/run18.log 2>&1