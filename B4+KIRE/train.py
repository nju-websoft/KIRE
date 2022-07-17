import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython
import yaml
import yamlordereddictloader
from knowledge_injection_layer.config import Config as kgconfig
# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'BiLSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)
parser.add_argument('--max_length', type = int, default= 512) # 1800
parser.add_argument('--dataset', type=str, default='docred')
parser.add_argument('--data_path', type=str, default='./prepro_data')  # './prepro_data/dwie'
parser.add_argument('--kg_data_path', type=str, default='../kg_data')  # ../DWIE/kg_data
parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--kgconfig', type=str, default='./knowledge_injection_layer/kgconfigs/base.yaml')


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
}
# kg 配置文件
with open(args.kgconfig, 'r', encoding="utf-8") as f:
    kg_params = yaml.load(f, Loader=yamlordereddictloader.Loader)
    for k,v in kg_params.items():
        setattr(kgconfig, k, v)
                    # print(kgconfig.gcn_layer_nums)
con = config.Config(args)
con.set_max_epoch(400)
con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)

# bilstm
# train
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
# test
# CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601

# bilstm_auto_gatrel
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name BiLSTM --save_name checkpoint_BiLSTM_auto_gatrel --train_prefix dev_train --test_prefix dev_dev --kgconfig ./knowledge_injection_layer/kgconfigs/kg_auto_gatrel.yaml
# test
# CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM_auto_gatrel --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601 --kgconfig ./knowledge_injection_layer/kgconfigs/kg_auto_gatrel.yaml

# bilstm_dwie
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name BiLSTM --save_name checkpoint_dwie_BiLSTM --train_prefix dev_train --test_prefix dev_dev --kgconfig ./knowledge_injection_layer/kgconfigs/kg_bilstm.yaml --max_length 1800 --dataset dwie --data_path ./prepro_data/dwie  --kg_data_path ../DWIE/kg_data
# test
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model_name BiLSTM --save_name checkpoint_dwie_BiLSTM_kere --train_prefix dev_train --test_prefix dev_dev --input_theta 0.6099 --kgconfig ./knowledge_injection_layer/kgconfigs/base.yaml --max_length 1800 --dataset dwie --data_path ./prepro_data/dwie --kg_data_path ../DWIE/kg_data
