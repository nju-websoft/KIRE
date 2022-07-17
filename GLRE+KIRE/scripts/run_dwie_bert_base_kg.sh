#! /bin/bash
#export CUDA_VISIBLE_DEVICES=$1

#if [! -d "./logs"]; then
#  mkdir logs
#fi

echo "dwie_bert_base_kg"
python run_dwie_bert_base_kg.py --batch=8 --gpu 1 >logs/train_dwie_bert_base_kg.log 2>&1 