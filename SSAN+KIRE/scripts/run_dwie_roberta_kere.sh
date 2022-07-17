#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1
mkdir results_bert_kere

python train.py --data_dir ../DWIE/data/ \
--dataset dwie \
--transformer_type roberta \
--model_name_or_path ../roberta_large/ \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 1 \
--test_batch_size 1 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 200.0 \
--seed 6 \
--num_class 66 \
--max_seq_length 2600 \
--save_path results_dwie_roberta_kere/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/dwie_roberta_kg.yaml \
>logs/train_dwie_roberta_large_kere.log 2>&1


python train.py --data_dir ../DWIE/data/ \
--dataset dwie \
--transformer_type roberta \
--model_name_or_path ../roberta_large/ \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 1 \
--test_batch_size 1 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 200.0 \
--seed 6 \
--num_class 66 \
--max_seq_length 2600 \
--load_path results_dwie_roberta_kere/re.model \
--kgfile ./knowledge_injection_layer/kgconfigs/dwie_roberta_kg.yaml \
>logs/test_dwie_roberta_large_kere.log 2>&1