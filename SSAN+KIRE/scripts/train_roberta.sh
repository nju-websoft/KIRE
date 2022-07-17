set -eux

export CUDA_VISIBLE_DEVICES=0
pretrained_model=./pretrained_lm/roberta_large/
data_dir=./data/DocRED/

lr=2e-5
epoch=40
batch_size=2

python run_docred.py \
  --dataset  docred\
  --model_type roberta \
  --entity_structure biaffine \
  --checkpoint_dir checkpoints/roberta_large \
  --model_name_or_path ${pretrained_model} \
  --do_train \
  --data_dir ${data_dir} \
  --max_seq_length 512 \
  --max_ent_cnt 42 \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --warmup_ratio 0.1 \
  --output_dir checkpoints/roberta_large \
  --seed 42 \
  --logging_steps 10 \
  --save_path results_roberta_kere/re_roberta_large.model \
  --kgfile ./knowledge_injection_layer/kgconfigs/roberta_kg.yaml \
  > lr=.log 2>&1
