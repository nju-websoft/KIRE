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
parser.add_argument('--model_name', type = str, default = 'LSTM', help = 'name of the model')
parser.add_argument('--save_name', type = str)
parser.add_argument('--max_length', type = int, default= 512) # 1800
parser.add_argument('--dataset', type=str, default='docred')
parser.add_argument('--data_path', type=str, default='./prepro_data')  # './prepro_data/dwie'
parser.add_argument('--kg_data_path', type=str, default='../kg_data')  # ../DWIE/kg_data
parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = -1)
parser.add_argument('--kgconfig', type=str, default='./knowledge_injection_layer/kgconfigs/base.yaml')
# parser.add_argument('--ignore_input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
	# 'LSTM_SP': models.LSTM_SP
}

# kg 配置文件
with open(args.kgconfig, 'r', encoding="utf-8") as f:
    kg_params = yaml.load(f, Loader=yamlordereddictloader.Loader)
    for k,v in kg_params.items():
        setattr(kgconfig, k, v)

con = config.Config(args)
#con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.testall(model[args.model_name], args.save_name, args.input_theta)#, args.ignore_input_theta)
