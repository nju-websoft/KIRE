
import os
import argparse
import sys
sys.path.append('../GLRE')
sys.path.append('GLRE')
sys.path.append('GLRE/knowledge_injection_layer')
from knowledge_injection_layer import config as kgconfig
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch', type=int, default=8)

inp = parser.parse_args()

config_path = './configs/dwie_basebert.yaml'
output_path = './results/dwie-dev/dwie_basebert/'
kg_config_path = './knowledge_injection_layer/kgconfigs/dwie_base.yaml'
os.system('CUDA_VISIBLE_DEVICES=' + str(inp.gpu)+ ' python ./main.py --train --batch=' + str(inp.batch)+ ' --test_data=./prepro_data/DWIE/processed/dev.data'
          ' --config_file=' + config_path+ ' --save_pred=dev --output_path=' + output_path + ' --kg_file=' + kg_config_path)

with open(os.path.join(output_path, "train_finsh.ok"), 'r') as f:
    for line in f.readlines():
        input_theta = line.strip().split("\t")[1]
        break

os.system('CUDA_VISIBLE_DEVICES=' + str(inp.gpu)+ ' python ./main.py --test --batch ' + str(inp.batch)+ ' --test_data ./prepro_data/DWIE/processed/dev.data'
          ' --config_file=' + config_path + ' --output_path=' + output_path
          + ' --save_pred=dev_test --input_theta='+str(input_theta) + ' --remodelfile='+output_path + ' --kg_file=' + kg_config_path)

os.system('CUDA_VISIBLE_DEVICES=' + str(inp.gpu)+ ' python ./main.py --test --batch ' + str(inp.batch)+ ' --test_data ./prepro_data/DWIE/processed/test.data'
          ' --config_file=' + config_path + ' --output_path=' + output_path
          + ' --save_pred=test --input_theta=' + str(input_theta) + ' --remodelfile=' + output_path + ' --kg_file=' + kg_config_path)

# os.system('python ./data_processing/convert2result.py --input_path='+ output_path
#            + 'test.errors --output_path=' + output_path + 'result.json')
