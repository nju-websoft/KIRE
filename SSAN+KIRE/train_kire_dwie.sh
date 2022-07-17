set -eux

pretrained_model=./pretrained_lm/bert-base-cased/
data_dir=./data/DWIE/
lr = 3e-5
epoch=200
batch_size=4

CUDA_VISIBLE_DEVICES=0 python ./run_docred.py \
  --dataset DWIE \
  --model_type bert \
  --entity_structure none \
  --model_name_or_path ${pretrained_model} \
  --do_train \
  --do_predict \
  --data_dir ${data_dir} \
  --max_seq_length 2500 \
  --max_ent_cnt 96 \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate 3e-5 \
  --num_train_epochs ${epoch} \
  --warmup_ratio 0.06 \
  --output_dir checkpoints/dwie_kire \
  --checkpoint_dir checkpoints/dwie_kire \
  --seed 66 \
  --logging_steps 10 \
  --kgfile ./knowledge_injection_layer/kgconfigs/dwie_bert_kg.yaml