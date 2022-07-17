cd ../../
CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name CNN3 \
--save_name checkpoint_CNN_DWIE_kere_1 \
--test_prefix dev_test \
--dataset dwie \
--max_length 1800 \
--kg_data_path ../DWIE/kg_data \
--data_path ./prepro_data/dwie \
--input_theta 0.3601 \
--kgconfig ./knowledge_injection_layer/kgconfigs/kg_bilstm.yaml 