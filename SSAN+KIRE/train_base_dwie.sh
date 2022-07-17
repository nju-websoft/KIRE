set -eux

export CUDA_VISIBLE_DEVICES=0
pretrained_model=./pretrained_lm/bert-base-cased/
data_dir=./data/DWIE/

lr=1e-5
epoch=40
batch_size=4

python run_docred.py \
  --dataset  DWIE\
  --model_type bert \
  --entity_structure none \
  --checkpoint_dir checkpoints/dwie \
  --model_name_or_path ${pretrained_model} \
  --do_train \
  --do_predict \
  --data_dir ${data_dir} \
  --max_seq_length 2500 \
  --max_ent_cnt 96 \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --warmup_ratio 0.1 \
  --output_dir checkpoints/dwie \
  --seed 42 \
  --logging_steps 10 \
  --kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_base.yaml 
