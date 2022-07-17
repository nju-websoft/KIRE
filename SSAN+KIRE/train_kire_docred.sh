set -eux

export CUDA_VISIBLE_DEVICES=0
pretrained_model=./pretrained_lm/bert-base-cased/
data_dir=./data/DocRED/

lr=5e-5
epoch=40
batch_size=4

python run_docred.py \
  --dataset  docred\
  --model_type bert \
  --entity_structure decomp \
  --checkpoint_dir checkpoints/bert_base \
  --model_name_or_path ${pretrained_model} \
  --do_train \
  --do_predict \
  --data_dir ${data_dir} \
  --max_seq_length 512 \
  --max_ent_cnt 42 \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --warmup_ratio 0.1 \
  --output_dir checkpoints/bert_base \
  --seed 42 \
  --logging_steps 10 \
  --kgfile ./knowledge_injection_layer/kgconfigs/bert_kg.yaml \