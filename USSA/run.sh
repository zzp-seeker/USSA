#!/bin/bash

# norec
python main.py --train_epochs 60 --batch_size 16 --learning_rate 2e-3 --dataset norec --td 0.7 --do_train 

# eu
python main.py --train_epochs 60 --batch_size 16 --learning_rate 2e-3 --dataset eu --td 1 --do_train 

# ca
python main.py --train_epochs 60 --batch_size 16 --learning_rate 1e-3 --dataset ca --td 0.7 --do_train  --max_len 386

# mpqa
python main.py --train_epochs 60 --batch_size 16 --learning_rate 2e-3 --dataset mpqa --td 0.55 --do_train --max_len 210

# ds
python main.py --train_epochs 60 --batch_size 16 --learning_rate 1e-3 --dataset ds --td 0.7 --do_train  --max_len 386