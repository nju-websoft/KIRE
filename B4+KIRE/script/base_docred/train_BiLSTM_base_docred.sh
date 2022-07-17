cd ../../
CUDA_VISIBLE_DEVICES=0 python train.py --model_name BiLSTM \
--save_name checkpoint_BiLSTM_DocRED \
--train_prefix dev_train \
--test_prefix dev_dev \
--dataset docred \
--max_length 512 \
--kg_data_path ../kg_data \
--data_path ./prepro_data \
--kgconfig ./knowledge_injection_layer/kgconfigs/base.yaml