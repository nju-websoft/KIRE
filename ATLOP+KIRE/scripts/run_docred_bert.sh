#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1
export DGLBACKEND=pytorch

mkdir results_bert
#python train.py --data_dir ../data/ \
#  --transformer_type bert \
#  --model_name_or_path ../bert_base/ \
#  --train_file train_annotated.json \
#  --dev_file dev.json \
#  --test_file test.json \
#  --train_batch_size 4 \
#  --test_batch_size 8 \
#  --gradient_accumulation_steps 1 \
#  --num_labels 4 \
#  --learning_rate 5e-5 \
#  --max_grad_norm 1.0 \
#  --warmup_ratio 0.06 \
#  --num_train_epochs 200.0 \
#  --seed 66 \
#  --num_class 97 \
#  --save_path results_bert/re.model \
#  --kgfile ./knowledge_injection_layer/kgconfigs/base.yaml \
#  >logs/train_bert_base.log 2>&1

python train.py --data_dir ../data/ \
  --dataset docred \
  --transformer_type bert \
  --model_name_or_path ../bert_base/ \
  --train_file train_annotated.json \
  --dev_file dev.json \
  --test_file test.json \
  --train_batch_size 4 \
  --test_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_labels 4 \
  --learning_rate 5e-5 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 200.0 \
  --seed 66 \
  --num_class 97 \
  --load_path results_bert/re.model \
  --kgfile ./knowledge_injection_layer/kgconfigs/base.yaml \
  >logs/test_bert_base.log 2>&1
