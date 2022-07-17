#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

from data.dataset import DocRelationDataset
from data.loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils.utils import setup_log, load_model, load_mappings
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 关闭TF的警告信息


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_train')
    set_seed(parameters['seed'])

    print('Loading training data ...')
    dataset_type = parameters['train_data'].split('/')[-1][:-5].split('_')[0]
    dataset_type = dataset_type.replace('1', '')
    train_loader = DataLoader(parameters['train_data'], parameters, dataset_type)
    # train_loader(embeds=parameters['embeds'], parameters=parameters)
    train_loader(embeds=None, parameters=parameters)
    train_data, _ = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()

    print('\nLoading testing data ...')
    dataset_type = parameters['test_data'].split('/')[-1][:-5].split('_')[0]
    dataset_type = dataset_type.replace('1', '')
    test_loader = DataLoader(parameters['test_data'], parameters, dataset_type, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    ###################################
    # Training
    ###################################
    trainer = Trainer(train_loader, parameters, {'train': train_data, 'test': test_data}, model_folder, prune_recall)

    trainer.run()


def _test(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_test')

    print('\nLoading mappings ...')
    dataset_type = parameters['train_data'].split('/')[-1][:-5].split('_')[0]
    dataset_type = dataset_type.replace('1', '')
    train_loader = DataLoader(parameters['train_data'], parameters, dataset_type)
    # train_loader(embeds=parameters['embeds'], parameters=parameters)
    train_loader(embeds=None, parameters=parameters)
    # train_loader = load_mappings(parameters['remodelfile'])
    
    print('\nLoading testing data ...')
    dataset_type = parameters['test_data'].split('/')[-1][:-5].split('_')[0]
    dataset_type = dataset_type.replace('1', '')
    test_loader = DataLoader(parameters['test_data'], parameters, dataset_type, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    m = Trainer(train_loader, parameters, {'train': [], 'test': test_data}, model_folder, prune_recall)
    trainer = load_model(parameters['remodelfile'], m, strict=False)
    trainer.eval_epoch(final=True, save_predictions=True)


def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        train(parameters)

    elif parameters['test']:
        _test(parameters)

if __name__ == "__main__":
    main()

