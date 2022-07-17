#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1
mkdir results_bert_kere_wo_attr
mkdir results_bert_kere_wo_coref
mkdir results_bert_kere_wo_corefloss
mkdir results_bert_kere_wo_kg
mkdir results_bert_kere_wo_kg_align_loss
mkdir results_bert_kere_wo_relation_module

#echo "run_docred_bert_base_kg_wo_kg_align_loss"
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
#--save_path results_bert_kere_wo_kg_align_loss/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_kg_align_loss.yaml \
#>logs/train_bert_base_kere_wo_kg_align_loss.log 2>&1
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
#--load_path results_bert_kere_wo_kg_align_loss/re.model \
#--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_kg_align_loss.yaml \
#>logs/test_bert_base_kere_wo_kg_align_loss.log 2>&1

echo "run_docred_bert_base_kg_wo_attr"
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
--save_path results_bert_kere_wo_attr/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_attr.yaml \
>logs/train_bert_base_kere_wo_attr.log 2>&1

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
--load_path results_bert_kere_wo_attr/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_attr.yaml \
>logs/test_bert_base_kere_wo_attr.log 2>&1

echo "run_docred_bert_base_kg_wo_coref"
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
--save_path results_bert_kere_wo_coref/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_coref.yaml \
  >logs/train_bert_base_kere_wo_coref.log 2>&1

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
--load_path results_bert_kere_wo_coref/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_coref.yaml \
  >logs/test_bert_base_kere_wo_coref.log 2>&1

echo "run_docred_bert_base_kg_wo_corefloss"
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
--save_path results_bert_kere_wo_corefloss/re.model \
  --kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_corefloss.yaml \
  >logs/train_bert_base_kere_wo_corefloss.log 2>&1

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
--load_path results_bert_kere_wo_corefloss/re.model \
  --kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_corefloss.yaml \
  >logs/test_bert_base_kere_wo_corefloss.log 2>&1

echo "run_docred_bert_base_kg_wo_kg"
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
--save_path results_bert_kere_wo_kg/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_kg.yaml \
>logs/train_bert_base_kere_wo_kg.log 2>&1

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
--load_path results_bert_kere_wo_kg/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_kg.yaml \
>logs/test_bert_base_kere_wo_kg.log 2>&1

echo "run_docred_bert_base_kg_wo_relation_module"
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
--save_path results_bert_kere_wo_relation_module/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_relation_module.yaml \
>logs/train_bert_base_kere_wo_relation_module.log 2>&1

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
--load_path results_bert_kere_wo_relation_module/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/bert_kg_wo_relation_module.yaml \
>logs/test_bert_base_kere_wo_relation_module.log 2>&1