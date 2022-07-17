#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1
export DGLBACKEND=pytorch

#python train.py --data_dir ../DWIE/data/ \
#--dataset dwie \
#--transformer_type roberta \
#--model_name_or_path ../roberta_large/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 1 \
#--test_batch_size 2 \
#--gradient_accumulation_steps 2 \
#--num_labels 4 \
#--learning_rate 3e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 66 \
#--max_seq_length 2500 \
#--save_path results_dwie_roberta/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/base.yaml \
#>logs/train_dwie_roberta_large.log 2>&1

python train.py --data_dir ../DWIE/data/ \
--dataset dwie \
--transformer_type roberta \
--model_name_or_path ../roberta_large/ \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 1 \
--test_batch_size 2 \
--gradient_accumulation_steps 2 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 200.0 \
--seed 66 \
--num_class 66 \
--max_seq_length 2500 \
--load_path results_dwie_roberta/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/base.yaml \
>logs/test_dwie_roberta_large.log 2>&1
