cd ../../
CUDA_VISIBLE_DEVICES=0 python train.py --model_name ContextAware \
--save_name checkpoint_ContextAware_DWIE \
--train_prefix dev_train \
--test_prefix dev_dev \
--dataset dwie \
--max_length 1800 \
--kg_data_path ../DWIE/kg_data \
--data_path ./prepro_data/dwie \
--kgconfig ./knowledge_injection_layer/kgconfigs/base.yaml