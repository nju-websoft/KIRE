set -eux

pretrained_model=./pretrained_lm/roberta_large/
data_dir=./data/DocRED/
checkpoint_dir=checkpoints
predict_thresh=0.46544307

CUDA_VISIBLE_DEVICES=0 python ./run_docred.py \
  --dataset docred \
  --model_type roberta \
  --entity_structure biaffine \
  --model_name_or_path ${pretrained_model} \
  --do_predict \
  --predict_thresh $predict_thresh \
  --data_dir ${data_dir} \
  --max_seq_length 512 \
  --max_ent_cnt 42 \
  --checkpoint_dir $checkpoint_dir \
  --seed 42 \
  --load_path results_roberta_kere/re.model \
  --kgfile ./knowledge_injection_layer/kgconfigs/bert_kg.yaml \
  >logs/test_roberta_large_kere.log 2>&1