cd ../../
CUDA_VISIBLE_DEVICES=1 python train.py --model_name CNN3 \
--save_name checkpoint_CNN_DWIE \
--train_prefix dev_train \
--test_prefix dev_dev \
--dataset dwie \
--max_length 1800 \
--kg_data_path ../DWIE/kg_data \
--data_path ./prepro_data/dwie \
--kgconfig ./knowledge_injection_layer/kgconfigs/kg_bilstm.yaml