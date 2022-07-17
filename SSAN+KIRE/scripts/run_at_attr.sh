#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1
mkdir results_bert_kere_at
mkdir results_bert_kere_at_onlyalias
mkdir results_bert_kere_at_onlydescription
mkdir results_bert_kere_at_onlyinstance
mkdir results_bert_kere_at_onlylabel

# 跑属性实验
#echo "运行run_docred_bert_base_kg_at"
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--save_path results_bert_kere_at/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at.yaml \
#>logs/train_bert_base_kere_at.log 2>&1
#
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--load_path results_bert_kere_at/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at.yaml \
#>logs/test_bert_base_kere_at.log 2>&1

#echo  "运行run_docred_bert_base_kg_at_onlylabel"
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--save_path results_bert_kere_at_onlylabel/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlylabel.yaml \
#>logs/train_bert_base_kere_at_onlylabel.log 2>&1
#
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--load_path results_bert_kere_at_onlylabel/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlylabel.yaml \
#>logs/test_bert_base_kere_at_onlylabel.log 2>&1


echo  "运行run_docred_bert_base_kg_at_onlydescription"
python train.py --data_dir ../data/ \
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
--save_path results_bert_kere_at_onlydescription/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlydescription.yaml \
>logs/train_bert_base_kere_at_onlydescription.log 2>&1

python train.py --data_dir ../data/ \
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
--load_path results_bert_kere_at_onlydescription/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlydescription.yaml \
>logs/test_bert_base_kere_at_onlydescription.log 2>&1

#echo  "运行run_docred_bert_base_kg_at_onlyinstance"
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--save_path results_bert_kere_at_onlyinstance/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlyinstance.yaml \
#>logs/train_bert_base_kere_at_onlyinstance.log 2>&1
#
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--load_path results_bert_kere_at_onlyinstance/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlyinstance.yaml \
#>logs/test_bert_base_kere_at_onlyinstance.log 2>&1
#
#echo  "运行run_docred_bert_base_kg_at_onlyalias"
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--save_path results_bert_kere_at_onlyalias/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlyalias.yaml \
#>logs/train_bert_base_kere_at_onlyalias.log 2>&1
#
#python train.py --data_dir ../data/ \
#--transformer_type bert \
#--model_name_or_path ../bert_base/ \
#--train_file train_annotated.json \
#--dev_file dev.json \
#--test_file test.json \
#--train_batch_size 4 \
#--test_batch_size 4 \
#--gradient_accumulation_steps 1 \
#--num_labels 4 \
#--learning_rate 5e-5 \
#--max_grad_norm 1.0 \
#--warmup_ratio 0.06 \
#--num_train_epochs 200.0 \
#--seed 66 \
#--num_class 97 \
#--load_path results_bert_kere_at_onlyalias/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_at_onlyalias.yaml \
#>logs/test_bert_base_kere_at_onlyalias.log 2>&1