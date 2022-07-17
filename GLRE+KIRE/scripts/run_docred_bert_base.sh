#! /bin/bash
#export CUDA_VISIBLE_DEVICES=$1

#if [! -d "./logs"]; then
#  mkdir logs
#fi

echo "docred_bert_base"
python run_docred_bert_base.py --batch=8 --gpu 1 >logs/train_docred_bert_base.log 2>&1