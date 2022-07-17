#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn
import sklearn.metrics

import torch
import numpy as np
import os
from time import time
import itertools
import copy
import datetime
import random

from transformers import AdamW, get_constant_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel

from models.glre import GLRE
from utils.adj_utils import convert_3dsparse_to_4dsparse
from utils.utils import print_results, write_preds, write_errors, print_options, save_model, load_model
from data.converter import concat_examples
from torch import optim, nn
from utils.metrics_util import Accuracy
from knowledge_injection_layer.config import Config as kgConfig
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

class Trainer:
    def __init__(self, loader, params, data, model_folder, prune_recall):
        """
        Trainer object.

        Args:
            loader: loader object that holds information for training data
            params (dict): model parameters
            data (dict): 'train' and 'test' data
        """
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.data = data
        self.test_prune_recall = prune_recall
        self.params = params
        self.rel_size = loader.n_rel
        self.loader = loader
        self.model_folder = model_folder

        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")
        self.gc = params['gc']
        self.epoch = params['epoch']
        self.es = params['early_stop']
        self.primary_metric = params['primary_metric']
        self.show_class = params['show_class']
        self.preds_file = os.path.join(model_folder, params['save_pred'])
        self.ok_file = os.path.join(model_folder, "train_finsh.ok")
        self.best_epoch = 0
        if 'optimizer' not in params:
            params['optimizer'] = 'adam'
        self.o_name = params['optimizer']

        self.train_res = {'loss': [], 'score': []}
        self.test_res = {'loss': [], 'score': []}

        # early-stopping
        if self.es:
            self.max_patience = self.params['patience']
            self.cur_patience = 0
            self.best_score = 0.0

        self.model = self.init_model()
        # if kgConfig.parallel_flag:
        #     self.optimizer, self.scheduler = self.set_optimizer(self.model.module)
        # else:
        self.optimizer, self.scheduler = self.set_optimizer(self.model)

        self.test_epoch = 1
        self.wikidatarel2id = {}
        with open('./knowledge_injection_layer/wikidatarel2id.txt') as f:
            for line in f.readlines():
                items = line.strip().split('\t')
                self.wikidatarel2id[items[0]] = int(items[1])
        self.id2wikidatarel = {v: k for k, v in self.wikidatarel2id.items()}

    def init_model(self):
        model_0 = GLRE(self.params, self.loader.pre_embeds,
                            sizes={'word_size': self.loader.n_words,
                                   'type_size': self.loader.n_type, 'rel_size': self.loader.n_rel},
                            maps={'word2idx': self.loader.word2index, 'idx2word': self.loader.index2word,
                                  'rel2idx': self.loader.rel2index, 'idx2rel': self.loader.index2rel,
                                  'type2idx': self.loader.type2index, 'idx2type': self.loader.index2type},
                            lab2ign=self.loader.label2ignore)

        # GPU/CPU
        if self.params['gpu'] != -1:
            model_0.to(self.device)

        # if kgConfig.parallel_flag:
        #     model_0 = DistributedDataParallel(model_0)

        return model_0

    def set_optimizer(self, model_0):
        # OPTIMIZER
        # do not regularize biases
        paramsbert = []  # lr=0 or 1e-5
        paramsbert0reg = []
        for p_name, p_value in model_0.word_embed.embedding.named_parameters():
            if not p_value.requires_grad:
                continue
            if '.bias' in p_name:
                paramsbert0reg.append(p_value)
            else:
                paramsbert.append(p_value)
        if kgConfig.add_kg_flag and kgConfig.attr_encode_type=='max':
            for p_name, p_value in model_0.kg_injection.kg_encoder.attr_encoder.word_embed.named_parameters():
                if not p_value.requires_grad:
                    continue
                if '.bias' in p_name:
                    paramsbert0reg.append(p_value)
                else:
                    paramsbert.append(p_value)
        if self.params['encoder'] != 'lstm':
            for p_n, p in model_0.pretrain_lm.named_parameters():
                if not p.requires_grad:
                    continue
                if '.bias' in p_n:
                    paramsbert0reg.append(p)
                else:
                    paramsbert.append(p)
        paramsbert_ids = list(map(id, paramsbert)) + list(map(id, paramsbert0reg))

        if kgConfig.add_kg_flag:
            paramskg = [param for p_name, param in model_0.kg_injection.named_parameters() if param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  # step 2
            paramskg0reg = [param for p_name, param in model_0.kg_injection.named_parameters() if param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  # step 2

            paramskg_ids = list(map(id, paramskg)) + list(map(id, paramskg0reg))
        else:
            paramskg = []
            paramskg0reg = []
            paramskg_ids = []

        if kgConfig.add_coref_flag:
            paramskg += [param for p_name, param in model_0.coref_injection.named_parameters() if
                         param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  # step 2
            paramskg0reg += [param for p_name, param in model_0.coref_injection.named_parameters() if
                             param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  # step 2
            paramskg_ids = list(map(id, paramskg)) + list(map(id, paramskg0reg))

        paramsothers = [param for p_name, param in model_0.named_parameters() if param.requires_grad and (id(param) not in (paramskg_ids+paramsbert_ids)) and '.bias' not in p_name]
        paramsothers0reg = [param for p_name, param in model_0.named_parameters() if param.requires_grad and (
                    id(param) not in (paramskg_ids + paramsbert_ids)) and '.bias' in p_name]
        groups = [dict(params=paramskg, lr=0.0), dict(params=paramsothers, lr=self.params['lr']), dict(params=paramsbert, lr=self.params['bert_lr']),
                  dict(params=paramskg0reg, lr=0.0, weight_decay=0.0), dict(params=paramsothers0reg, lr=self.params['lr'], weight_decay=0.0),
                  dict(params=paramsbert0reg, lr=self.params['bert_lr'], weight_decay=0.0)]

        scheduler = None
        if self.o_name == 'adam':
            optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=0, amsgrad=False)
            # optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=float(self.params['reg']), amsgrad=True)
        elif self.o_name == 'adamw':
            #optimizer = AdamW(groups, lr=self.params['lr'], weight_decay=float(self.params['reg']), correct_bias=False)
            optimizer = AdamW(groups, lr=self.params['lr'], weight_decay=0, correct_bias=False)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2000)


        # Train Model
        print_options(self.params)
        for p_name, p_value in model_0.named_parameters():
            if p_value.requires_grad:
                print(p_name)

        for k, v in kgConfig.__dict__.items():
            if not str(k).startswith('__'):
                print(k, '\t', v)
        return optimizer, scheduler

    @staticmethod
    def iterator(x, shuffle_=False, batch_size=1):
        """
        Create a new iterator for this epoch.
        Shuffle the data if specified.
        """
        if shuffle_:
            random.shuffle(x)
        new = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
        return new

    def run(self):
        """
        Main Training Loop.
        """
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

        random.shuffle(self.data['train'])  # shuffle training data at least once
        best_dev_f1 = -1.0
        final_best_theta = 0.5

        init_add_kg_flag = kgConfig.add_kg_flag
        init_add_coref_flag = kgConfig.add_coref_flag
        if kgConfig.train_method in ['two_step', 'three_step']:
            kgConfig.add_kg_flag = False
            kgConfig.add_coref_flag = False
            train_step = 0
        else:
            self.optimizer.param_groups[0]['lr'] = self.params['lr']
            self.optimizer.param_groups[3]['lr'] = self.params['lr']
            train_step = 2
        self.cur_patience = 0
        # todo if 继续训练， 把train_step = 1即可
        for epoch in range(1, self.epoch + 1):

            if kgConfig.train_method == 'one_step' and kgConfig.re_train:
                load_model(kgConfig.re_train_path, self, False, ignore=['kg_linear_transfer.0.weight', 'kg_linear_transfer.2.weight',
                                                                        'kg_linear_transfer.0.bias','kg_linear_transfer.2.bias'])
                dev_f1, zj_f1, theta = self.eval_epoch()

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    final_best_theta = theta
                    print("dev f1=%f, save model" % dev_f1)
                    best_epoch = 0
                    save_model(self.model_folder, self, self.loader)
                break
            if kgConfig.re_train and train_step == 0:
                train_step = 1 # 跳过第一部分训练
                self.params['init_train_epochs'] = 0

            if kgConfig.train_method == 'two_step' and train_step == 1:
                print("加载最优模型")
                if kgConfig.re_train:  # 加载之前训练好的模型
                    load_model(kgConfig.re_train_path, self, False, ['word_embed.embedding.weight'])
                else:
                    load_model(self.model_folder, self)
                if init_add_kg_flag:
                    kgConfig.add_kg_flag = True
                if init_add_coref_flag:
                    kgConfig.add_coref_flag = True
                self.optimizer.param_groups[0]['lr'] = kgConfig.kg_lr
                self.optimizer.param_groups[1]['lr'] = kgConfig.other_lr
                self.optimizer.param_groups[3]['lr'] = kgConfig.kg_lr
                self.optimizer.param_groups[4]['lr'] = kgConfig.other_lr
                train_step = 2
                self.best_score = 0  # 重新开始计算
                best_dev_f1 = 0
                self.max_patience = 20

            if kgConfig.train_method == 'three_step' and train_step == 1:
                print("加载最优模型")
                load_model(self.model_folder, self)
                if init_add_kg_flag:
                    kgConfig.add_kg_flag = True
                if init_add_coref_flag:
                    kgConfig.add_coref_flag = True
                self.optimizer.param_groups[0]['lr'] = kgConfig.kg_lr
                self.optimizer.param_groups[1]['lr'] = 0.0
                self.optimizer.param_groups[3]['lr'] = kgConfig.kg_lr
                self.optimizer.param_groups[4]['lr'] = 0.0
                train_step = 2
                self.best_score = 0
                best_dev_f1 = 0

            if kgConfig.train_method == 'three_step' and train_step == 3:
                print("加载最优模型")
                load_model(self.model_folder, self)
                if init_add_kg_flag:
                    kgConfig.add_kg_flag = True
                if init_add_coref_flag:
                    kgConfig.add_coref_flag = True
                self.optimizer.param_groups[0]['lr'] = kgConfig.other_lr
                self.optimizer.param_groups[1]['lr'] = kgConfig.other_lr
                self.optimizer.param_groups[3]['lr'] = kgConfig.other_lr
                self.optimizer.param_groups[4]['lr'] = kgConfig.other_lr
                train_step = 4
                self.best_score = 0
                best_dev_f1 = 0

            train_f1 = self.train_epoch(epoch)

            if epoch < self.params['init_train_epochs']:
                continue


            if epoch % self.test_epoch == 0:

                dev_f1, zj_f1, theta = self.eval_epoch()

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    final_best_theta = theta
                    best_train_f1 = train_f1
                    print("dev f1=%f, save model" % dev_f1)
                    save_model(self.model_folder, self, self.loader)

            if self.es:
                best_epoch, stop = self.early_stopping(epoch)
                if stop:
                    if train_step == 0:
                        train_step = 1
                        self.cur_patience = 0
                    else:
                        if kgConfig.train_method == 'three_step' and train_step == 2:
                            train_step = 3
                            self.cur_patience = 0
                        else:
                            break

        if self.es and (epoch != self.epoch):
            print('Best epoch: {}'.format(best_epoch))
            self.eval_epoch(final=True, save_predictions=True)
            self.best_epoch = best_epoch

        elif epoch == self.epoch:

            self.eval_epoch(final=True, save_predictions=True)

        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))
        with open(self.ok_file, "w") as f:
            f.write(str(best_dev_f1) + '\t' + str(final_best_theta))

    def train_epoch(self, epoch):
        """
        Evaluate the model on the train set.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'true': 0, 'ttotal': [], 're_loss':[] ,'coref_loss':[], 'kg_loss': []}
        self.acc_NA.clear()
        self.acc_not_NA.clear()
        self.acc_total.clear()

        # self.model.train()
        train_iter = self.iterator(self.data['train'], batch_size=self.params['batch'],
                                   shuffle_=self.params['shuffle_data'])
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_iter):  # todo 将联合训练 改为 交替训练
            # t1 = time()
            self.model.train()
            batch = self.convert_batch(batch, istrain=True)
            # t2 = time()
            # with autograd.detect_anomaly():
            # self.optimizer.zero_grad()
            loss, stats, predictions, select, pred_pairs, multi_truths, mask, relation_label, loss_re, loss_coref, loss_kg, _ = self.model(batch)
            pred_pairs = torch.sigmoid(pred_pairs)
            # t3 = time()
            # self.optimizer.zero_grad()
            loss = loss / kgConfig.gradient_accumulation_steps

            loss.backward()
            # t4 = time()
            # print("外部时间", t2-t1, t3-t2, t4-t3) # 0.19, 2.95, 3.8
            if batch_idx % kgConfig.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)  # gradient clipping
                self.optimizer.step()  # update
                if self.scheduler!=None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            relation_label = relation_label.to('cpu').data.numpy()
            predictions = predictions.to('cpu').data.numpy()
            output['loss'] += [float(loss.item())]
            if loss_coref is not None:
                output['coref_loss'] += [float(loss_coref.item())]
            else:
                output['coref_loss'] += [0.0]
            if loss_kg is not None:
                output['kg_loss'] += [float(loss_kg.item())]
            else:
                output['kg_loss'] += [0.0]
            output['re_loss'] += [float(loss_re.item())]
            output['tp'] += [stats['tp'].to('cpu').data.numpy()]
            output['fp'] += [stats['fp'].to('cpu').data.numpy()]
            output['fn'] += [stats['fn'].to('cpu').data.numpy()]
            output['tn'] += [stats['tn'].to('cpu').data.numpy()]
            output['preds'] += [predictions]
            output['ttotal'] += [stats['ttotal']]

            for i in range(predictions.shape[0]):
                label = relation_label[i]
                if label < 0:
                    break
                assert self.loader.label2ignore == 0
                if label == self.loader.label2ignore:
                    self.acc_NA.add(predictions[i] == label)
                else:
                    self.acc_not_NA.add(predictions[i] == label)
                self.acc_total.add(predictions[i] == label)

        total_loss, scores, total_re_loss, total_coref_loss, total_kg_loss = self.performance(output)
        t2 = time()

        self.train_res['loss'] += [total_loss]
        self.train_res['score'] += [scores[self.primary_metric]]
        print('Epoch: {:02d} | TRAIN | LOSS = {:.05f}, RE_LOSS = {:.05f},COREF_LOSS = {:.05f}, KG_LOSS = {:.05f},| NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f}'.
              format(epoch, total_loss, total_re_loss, total_coref_loss, total_kg_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()), end="")
        print_results(scores, [], self.show_class, t2 - t1)
        print("TTotal\t", sum(output['ttotal']))

        return scores['micro_f']

    def eval_epoch(self, final=False, save_predictions=False):
        """
        Evaluate the model on the test set.
        No backward computation is allowed.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'true': 0, 'kg_loss': [], 'coref_loss': [], 're_loss': []}
        test_info = []
        test_result = []
        self.model.eval()
        test_iter = self.iterator(self.data['test'], batch_size=self.params['batch'], shuffle_=False)
        for batch_idx, batch in enumerate(test_iter):
            batch = self.convert_batch(batch, istrain=False, save=True)

            with torch.no_grad():
                loss, stats, predictions, select, pred_pairs, multi_truths, mask, _, re_loss, _, _, preds = self.model(
                    batch)  # pred_pairs <#pair, relations_num>
                pred_pairs = torch.sigmoid(pred_pairs)

                output['loss'] += [loss.item()]
                output['re_loss'] += [re_loss.item()]
                output['coref_loss'] += [0.0]
                output['kg_loss'] += [0.0]
                output['tp'] += [stats['tp'].to('cpu').data.numpy()]
                output['fp'] += [stats['fp'].to('cpu').data.numpy()]
                output['fn'] += [stats['fn'].to('cpu').data.numpy()]
                output['tn'] += [stats['tn'].to('cpu').data.numpy()]
                output['preds'] += [predictions.to('cpu').data.numpy()]

                if True:
                    test_infos = batch['info'][select[0].to('cpu').data.numpy(),
                                               select[1].to('cpu').data.numpy(),
                                               select[2].to('cpu').data.numpy()][mask.to('cpu').data.numpy()]
                    test_info += [test_infos]

            pred_pairs = pred_pairs.data.cpu().numpy()
            if self.params['adaptive_threshlod']:
                preds = preds.data.cpu().numpy()
            multi_truths = multi_truths.data.cpu().numpy()
            output['true'] += multi_truths.sum() - multi_truths[:, self.loader.label2ignore].sum()
            if save_predictions:
                assert test_infos.shape[0] == len(pred_pairs), print(
                    "test info=%d, pred_pair=%d" % (len(test_infos.shape[0]), len(pred_pairs)))

            if self.params['adaptive_threshlod']:
                for pair_id in range(len(preds)):
                    multi_truth = multi_truths[pair_id]
                    for r in range(0, self.rel_size):
                        if r == self.loader.label2ignore:
                            continue
                        test_result.append((int(multi_truth[r]) == 1, float(preds[pair_id][r]),
                                            test_infos[pair_id]['intrain'], test_infos[pair_id]['cross'],
                                            self.loader.index2rel[r], r,
                                            len(test_info) - 1, pair_id))
            else:
                for pair_id in range(len(pred_pairs)):
                    multi_truth = multi_truths[pair_id]
                    for r in range(0, self.rel_size):
                        if r == self.loader.label2ignore:
                            continue

                        test_result.append((int(multi_truth[r]) == 1, float(pred_pairs[pair_id][r]),
                                            test_infos[pair_id]['intrain'],test_infos[pair_id]['cross'], self.loader.index2rel[r], r,
                                            len(test_info) - 1, pair_id))


        # estimate performance
        total_loss, scores,_, _, _ = self.performance(output)

        test_result.sort(key=lambda x: x[1], reverse=True)
        if self.params['adaptive_threshlod']:
            input_theta, w, f1 = self.tune_f1_theta(test_result, output['true'], 0.0,
                                                    isTest=save_predictions)
        else:
            input_theta, w, f1 = self.tune_f1_theta(test_result, output['true'], self.params['input_theta'], isTest=save_predictions)

        # 计算各类F1
        # test_result = test_result[ :w+1]
        # self.macro_f1_cal(test_result, total_recall={"r1": 100, "r2":100})

        t2 = time()
        if not final:
            self.test_res['loss'] += [total_loss]
            # self.test_res['score'] += [scores[self.primary_metric]]
            self.test_res['score'] += [f1]
        # print('            TEST  | LOSS = {:.05f}, '.format(total_loss), end="")
        # print_results(scores, [], self.show_class, t2 - t1)
        print()

        if save_predictions:

            test_result = test_result[: w + 1]
            # test_result = test_result[:15000]
            test_result_pred = []
            test_result_info = []
            for item in test_result:
                test_result_pred.append([(item[-3], item[1])])
                test_result_info.append([test_info[item[-2]][item[-1]]])
                assert (item[-3] in test_info[item[-2]][item[-1]]['rel']) == item[0], print("item\n", item, "\n",
                                                                                            test_info[item[-2]][
                                                                                                item[-1]])
            write_errors(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel, type="theta")
            write_preds(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel)

        return f1, scores['micro_f'], input_theta

    def early_stopping(self, epoch):
        """
        Perform early stopping.
        If performance does not improve for a number of consecutive epochs ("max_patience")
        then stop the training and keep the best epoch: stopped_epoch - max_patience

        Args:
            epoch (int): current training epoch

        Returns: (int) best_epoch, (bool) stop
        """
        if len(self.test_res['score']) == 0:
            return -1, False
        if self.test_res['score'][-1] > self.best_score:  # improvement
            self.best_score = self.test_res['score'][-1]
            self.cur_patience = 0
        else:
            self.cur_patience += 1

        if self.max_patience == self.cur_patience:  # early stop must happen
            best_epoch = epoch - self.max_patience
            return best_epoch, True
        else:
            return epoch, False

    @staticmethod
    def performance(stats):
        """
        Estimate total loss for an epoch.
        Calculate Micro and Macro P/R/F1 scores & Accuracy.
        Returns: (float) average loss, (float) micro and macro P/R/F1
        """

        def fbeta_score(precision, recall, beta=1.0):
            beta_square = beta * beta
            if (precision != 0.0) and (recall != 0.0):
                res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
            else:
                res = 0.0
            return res

        def prf1(tp_, fp_, fn_, tn_):
            tp_ = np.sum(tp_, axis=0)
            fp_ = np.sum(fp_, axis=0)
            fn_ = np.sum(fn_, axis=0)
            tn_ = np.sum(tn_, axis=0)

            atp = np.sum(tp_)
            afp = np.sum(fp_)
            afn = np.sum(fn_)
            atn = np.sum(tn_)

            micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
            micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
            micro_f = fbeta_score(micro_p, micro_r)

            pp = [0]
            rr = [0]
            ff = [0]
            macro_p = np.mean(pp)
            macro_r = np.mean(rr)
            macro_f = np.mean(ff)

            acc = (atp + atn) / (atp + atn + afp + afn) if (atp + atn + afp + afn) else 0.0
            acc_NA = atn / (atn + afp) if (atn + afp) else 0.0
            acc_not_NA = atp / (atp + afn) if (atp + afn) else 0.0
            return {'acc': acc, 'NA_acc': acc_NA, 'not_NA_acc': acc_not_NA,
                    'micro_p': micro_p, 'micro_r': micro_r, 'micro_f': micro_f,
                    'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f,
                    'tp': atp, 'true': atp + afn, 'pred': atp + afp, 'total': (atp + atn + afp + afn)}

        fin_loss = sum(stats['loss']) / len(stats['loss'])
        fin_re_loss = sum(stats['re_loss']) / len(stats['re_loss'])
        fin_coref_loss = sum(stats['coref_loss']) / len(stats['coref_loss'])
        fin_kg_loss = sum(stats['kg_loss']) / len(stats['kg_loss'])
        scores = prf1(stats['tp'], stats['fp'], stats['fn'], stats['tn'])
        return fin_loss, scores, fin_re_loss, fin_coref_loss, fin_kg_loss

    def macro_f1_cal(self, test_result, total_recall):
        """

        :param test_result:
        :param total_recall: {“r1”: 100, "r2": 20}
        :return:
        """
        pred = {} # 各关系预测个数 TP + FP
        correct = {}  # 各关系预测正确个数 TP
        for key in total_recall.keys():
            pred[key] = 0
            correct[key] = 0
        for i, item in enumerate(test_result):
            pred_rel = item[5]
            pred[pred_rel] += 1
            if item[0]:
                correct[pred_rel] += 1
        for key, recall in total_recall:
            p = correct[key] / pred[key]
            r = correct[key] / recall
            f1 = 2*p*r / (p+r)
            print("关系%s, P=%f, R=%f, F1=%f" % (key, p, r, f1))

    def tune_f1_theta(self, test_result, total_recall, input_theta=-1, isTest=False):
        """
        (truth==r, float(pred_pairs[pair_id][r]), test_info[pair_id]['intrain'], self.loader.index2rel(r),
        test_info[pair_id]['entA'].id, test_info[pair_id]['entB'].id, r)
        @:param test_result
        :return:
        """
        print("total_recall", total_recall)  # dev=12323
        # assert total_recall == self.test_prune_recall['0-max'], print(self.test_prune_recall['0-max'])
        if total_recall != self.test_prune_recall['0-max']:
            print(total_recall, self.test_prune_recall['0-max'])

        if total_recall == 0:
            total_recall = 1  # for test
        pr_x = []
        pr_y = []
        correct = 0
        w = 0
        for i, item in enumerate(test_result):
            # print("test_result", item)
                correct += item[0]
                pr_y.append(float(correct) / (i + 1))
                pr_x.append(float(correct) / total_recall)  ## R
                if item[1] > input_theta:
                    w = i
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        print('tune theta | ma_F1 {:3.4f} | ma_P {:3.4f} | ma_R {:3.4f} | AUC {:3.4f}'.format(f1, pr_y[f1_pos], pr_x[f1_pos], auc))
        print('input_theta {:3.4f} | test_result | F1 {:3.4f} | P {:3.4f} | R {:3.4f}'.format(input_theta, f1_arr[w], pr_y[w], pr_x[w]))
        f1_all = f1

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & (item[2].lower() == 'true'):
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
                .format(f1, input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))

        if isTest:
            print(self.test_prune_recall)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-1'], input_theta, 0, 1)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-3'], input_theta, 1, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-3'], input_theta, 0, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-max'], input_theta, 1, 10000)
            self.prune_f1_cal(test_result, self.test_prune_recall['3-max'], input_theta, 3, 10000)

        return input_theta, w, f1_all

    def prune_f1_cal(self, test_result, total_recall, input_theta, prune_k_s, prune_k_e):
        if total_recall == 0:
            return
        pr_x = []
        pr_y = []
        correct_in_prune_k = all_in_prune_k = 0
        w = 0
        j = 0
        for i, item in enumerate(test_result):
            dis = int(item[3])
            if dis>=prune_k_s and dis < prune_k_e:
                all_in_prune_k += 1
                j += 1
                if item[0]:
                    correct_in_prune_k += 1
                pr_y.append(float(correct_in_prune_k) / all_in_prune_k)
                pr_x.append(float(correct_in_prune_k) / total_recall)
                if item[1] > input_theta:
                    w = j
        if len(pr_x) == 0:
            print('prune {:1f}-{:2f} no value'.format(prune_k_s, prune_k_e))
            return
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print(
            'prune {}-{} | ma_f1 {:3.4f} | ma_p {:3.4f} | ma_r {:3.4f}| input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
            .format(prune_k_s, prune_k_e, f1, pr_y[f1_pos], pr_x[f1_pos], input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))

        pr_x = []
        pr_y = []
        correct_in_prune_k = correct_in_train = 0
        all_in_prune_k = 0
        w = 0
        j = 0
        for i, item in enumerate(test_result):
            dis = int(item[3])
            if dis >= prune_k_s and dis < prune_k_e:
                all_in_prune_k+=1
                j+=1
                correct_in_prune_k += item[0]
                if item[0] & (item[2].lower() == 'true'):
                    correct_in_train += 1
                if correct_in_train == correct_in_prune_k:
                    p = 0
                else:
                    p = float(correct_in_prune_k - correct_in_train) / (all_in_prune_k - correct_in_train)
                pr_y.append(p)
                pr_x.append(float(correct_in_prune_k) / total_recall)
                if item[1] > input_theta:
                    w = j
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print(
            'Ignore prune {}-{} | ma_f1 {:3.4f} | input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
            .format(prune_k_s, prune_k_e, f1, input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))

    def convert_batch(self, batch, istrain=False, save=False):
        new_batch = {'entities': [], 'bert_token': [], 'bert_mask': [], 'bert_starts': [], 'pos_idx': []}
        ent_count, sent_count, word_count = 0, 0, 0
        full_text = []
        # max_length = 512
        batch_size = len(batch)
        # max_coref_mention_size = 250
        # max_pair_cnt = 3200

        for i, b in enumerate(batch):
            current_text = list(itertools.chain.from_iterable(b['text']))
            full_text += current_text
            new_batch['bert_token'] += [b['bert_token']]
            new_batch['bert_mask'] += [b['bert_mask']]
            new_batch['bert_starts'] += [b['bert_starts']]

            temp = []
            for e in b['ents']:
                temp += [[e[0] + ent_count, e[1], e[2] + word_count, e[3] + word_count, e[4] + sent_count, e[4], e[5]]]  # id  type start end sent_id

            new_batch['entities'] += [np.array(temp)]
            word_count += sum([len(s) for s in b['text']])
            ent_count = max([t[0] for t in temp]) + 1
            sent_count += len(b['text'])

        new_batch['entities'] = torch.as_tensor(np.concatenate(new_batch['entities'], axis=0)).long().to(self.device)
        new_batch['bert_token'] = torch.as_tensor(np.concatenate(new_batch['bert_token'])).long().to(self.device)
        new_batch['bert_mask'] = torch.as_tensor(np.concatenate(new_batch['bert_mask'])).long().to(self.device)
        new_batch['bert_starts'] = torch.as_tensor(np.concatenate(new_batch['bert_starts'])).long().to(self.device)

        convert_items = ['adjacency', 'dist_dir', 'rels', 'multi_rels', 'section', 'word_sec', 'words', 'ners', 'coref_pos']
        batch_ = [{k: v for k, v in b.items() if k in convert_items} for b in batch]

        converted_batch = concat_examples(batch_, device=self.device, padding={key: 0 if key in ["multi_rels", "adjacency", "dist_dir"] else -1 for key in convert_items})

        new_batch['adjacency'] = converted_batch['adjacency'].float()  # 2,71,71
        new_batch['distances_dir'] = converted_batch['dist_dir'].long()  # 2,71,71
        new_batch['relations'] = converted_batch['rels'].float()
        new_batch['multi_relations'] = converted_batch['multi_rels'].float().clone()
        if istrain and self.params['NA_NUM'] < 1.0:
            NA_id = self.loader.label2ignore
            index = new_batch['multi_relations'][:, :, :, NA_id].nonzero()
            if index.size(0)!=0:
                value = (torch.rand(len(index)) < self.params['NA_NUM']).float()
                if (value == 0).all():
                    value[0] = 1.0
                value = value.to(self.device)
                new_batch['multi_relations'][index[:, 0], index[:, 1], index[:, 2], NA_id] = value

        new_batch['section'] = converted_batch['section'].long()  # 2, 4
        new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec'] != -1].long()  # 21
        new_batch['words'] = converted_batch['words'][converted_batch['words'] != -1].long().contiguous()  # 382
        new_batch['rgcn_adjacency'] = convert_3dsparse_to_4dsparse([b['rgcn_adjacency'] for b in batch]).to(self.device)
        new_batch['ners'] = converted_batch['ners'][converted_batch['ners'] != -1].long().contiguous()
        new_batch['coref_pos'] = converted_batch['coref_pos'][converted_batch['coref_pos'] != -1].long().contiguous()
        batch_max_length = new_batch['section'][:, -1].max()

        # Knowledge Injection 相关数据
        if self.params['dataset'] == 'docred':
            max_attr_len = 128
            max_entity_size = 42
            max_attr_size = 12000  # 一篇doc中最多的属性个数
            if batch_size <= 2:
                max_attr_size = 20000
            elif batch_size <= 4:
                max_attr_size = 15000
            max_length = 512
            max_coref_mention_size = 250
            max_pair_cnt = 3200
        elif self.params['dataset'] == 'dwie':
            max_attr_len = 128
            max_entity_size = 100
            max_attr_size = 15000  # 一篇doc中最多的属性个数
            if batch_size <= 2:
                max_attr_size = 45000
            elif batch_size <= 4:
                max_attr_size = 45000
            max_length = 1800
            max_coref_mention_size = 800
            max_pair_cnt = 9000


        kg_ent_adj = []
        kg_ent_radj = []
        kg_ent_adj_edges = []
        kg_ent_mask = torch.LongTensor(batch_size, max_length).cpu()
        kg_ent_labels = torch.LongTensor(batch_size, batch_max_length).cpu()
        coref_h_mapping = torch.FloatTensor(batch_size, max_pair_cnt, max_length).cpu()
        coref_t_mapping = torch.FloatTensor(batch_size, max_pair_cnt, max_length).cpu()
        coref_lens = torch.LongTensor(batch_size, max_coref_mention_size).cpu()
        coref_label = torch.FloatTensor(batch_size, max_pair_cnt, 2).cpu()
        coref_label_mask = torch.BoolTensor(batch_size, max_pair_cnt).cpu()
        coref_mention_position = torch.FloatTensor(batch_size, max_coref_mention_size, max_length).cpu()
        kg_ent_attrs = torch.LongTensor(batch_size*max_attr_size, max_attr_len)
        kg_ent_attr_nums = []
        kg_ent_attr_lens = torch.LongTensor(batch_size*max_attr_size)
        kg_ent_attr_indexs = np.zeros((batch_size, max_entity_size), dtype=object)

        IGNORE_INDEX = -100
        kg_ent_labels.fill_(IGNORE_INDEX)
        coref_label_mask.fill_(False)
        for k in [kg_ent_mask, coref_h_mapping, coref_t_mapping, coref_lens, coref_label, coref_mention_position, kg_ent_attrs, kg_ent_attr_lens]:
            k.zero_()

        max_coref_pair = 0
        attr_index = 0
        max_ent_adj_nums = max([int(b['kg_ent_adj_nums']) for b in batch])
        for i, b in enumerate(batch):
            def _convert2sparse(adj, shape, entity_node_size):
                """

                :param adj: [h,t,r]
                :param entity_node_size: 实体节点个数，移除需要预测的实体对之间的边
                :return:
                """
                # print("entity_node_size", entity_node_size)
                visit = set()
                row_ids = []
                col_ids = []
                vals = []
                edges = []
                for h,t,r in adj:
                    if kgConfig.relation_type != 'all':
                        if kgConfig.relation_type != r:
                            continue
                    if ((h,t) not in visit) and ( h==t or h >= entity_node_size or t >= entity_node_size):
                        visit.add((h, t))
                        row_ids.append(h)
                        col_ids.append(t)
                        vals.append(1.0)
                        edges.append(r)
                indices = torch.from_numpy(np.vstack((row_ids, col_ids)).astype(np.int64))
                values = torch.FloatTensor(vals)
                edges = torch.LongTensor(edges)
                return torch.sparse.FloatTensor(indices, values, shape), torch.sparse.LongTensor(indices, edges, shape)

            _ent_adj, _ent_edge = _convert2sparse(b['kg_ent_adj'], shape=(max_ent_adj_nums, max_ent_adj_nums), entity_node_size=len(b['kg_ent_nids']))
            # 获取rgcn 输入
            _ent_radj = [None for i in range(len(self.wikidatarel2id.keys()))]
            for r, r_index in self.wikidatarel2id.items():
                __adj = [(_h, _t, _r) for _h, _t, _r in b['kg_ent_adj'] if _r == r]
                _ent_radj[r_index] = _convert2sparse(__adj, shape=(max_ent_adj_nums, max_ent_adj_nums),
                                                     entity_node_size=len(b['kg_ent_nids']))[0].cuda()

            kg_ent_radj.append(_ent_radj)
            kg_ent_adj.append(_ent_adj.to(self.device))  # ent_adj padding到最大节点
            kg_ent_adj_edges.append(_ent_edge.to(self.device))
            kg_ent_mask[i].copy_(torch.from_numpy(b['kg_ent_mask']))
            # 聚合candicate
            # for j, nid in enumerate(b['kg_ent_nids']):  # 遍历各实体对应的nid
            #     for index in b['kg_ent_attr_indexs'][j]:  #
            #         kg_ent_labels[i, index] = nid
            attr_nums = copy.deepcopy(b['kg_ent_attr_nums'])
            # print(kg_ent_attrs[attr_index: attr_index + sum(attr_nums)].shape)
            # kg_ent_attrs[attr_index: attr_index + sum(attr_nums)].copy_(torch.from_numpy(b['kg_ent_attrs'][:sum(attr_nums)]))
            # kg_ent_attr_lens[attr_index: attr_index + sum(attr_nums)].copy_(torch.from_numpy(b['kg_ent_attr_lens'][:sum(attr_nums)]))
            kg_ent_attrs[attr_index: attr_index + sum(attr_nums)].copy_(
                torch.from_numpy(b['kg_ent_attrs'][:len(kg_ent_attrs[attr_index: attr_index + sum(attr_nums)])]))
            kg_ent_attr_lens[attr_index: attr_index + sum(attr_nums)].copy_(torch.from_numpy(b['kg_ent_attr_lens'][:len(kg_ent_attr_lens[attr_index: attr_index + sum(attr_nums)])]))
            kg_ent_attr_nums.append(attr_nums)
            kg_ent_attr_indexs[i] = b['kg_ent_attr_indexs']
            attr_index += sum(attr_nums)

            coref_h_mapping[i].copy_(torch.from_numpy(b['coref_h_mapping'].todense()))
            coref_t_mapping[i].copy_(torch.from_numpy(b['coref_t_mapping'].todense()))
            coref_label[i].copy_(torch.from_numpy(b['coref_label']))
            coref_lens[i].copy_(torch.from_numpy(b['coref_lens']))
            coref_label_mask[i].copy_(torch.from_numpy(b['coref_label_mask']))
            coref_mention_position[i].copy_(torch.from_numpy(b['coref_mention_position']))
            max_coref_pair = max(max_coref_pair, int(coref_label_mask[i].sum().item()))

        new_batch['kg_ent_mask'] = kg_ent_mask[:, :batch_max_length].contiguous().to(self.device)
        new_batch['kg_ent_attrs'] = kg_ent_attrs[:attr_index].contiguous().to(self.device)
        new_batch['kg_ent_attr_lens'] = kg_ent_attr_lens[:attr_index].contiguous()  # 放置到CPU即可
        new_batch['kg_ent_attr_nums'] = kg_ent_attr_nums
        new_batch['kg_ent_attr_indexs'] = kg_ent_attr_indexs
        new_batch['kg_ent_labels'] = kg_ent_labels.to(self.device)
        new_batch['kg_adj'] = kg_ent_adj  # dev_420
        new_batch['kg_radj'] = kg_ent_radj
        new_batch['kg_adj_edges'] = kg_ent_adj_edges

        new_batch['coref_h_mapping'] = coref_h_mapping[:, :max_coref_pair, :batch_max_length].contiguous().to(self.device)
        new_batch['coref_t_mapping'] = coref_t_mapping[:, :max_coref_pair, :batch_max_length].contiguous().to(self.device)
        new_batch['coref_lens'] = coref_lens[:, :].contiguous().to(self.device)
        new_batch['coref_label'] = coref_label[:, :max_coref_pair].contiguous().to(self.device)
        new_batch['coref_label_mask'] = coref_label_mask[:, :max_coref_pair].contiguous().to(self.device)
        new_batch['coref_mention_position'] = coref_mention_position[:, :, :batch_max_length].contiguous().to(self.device)

        assert any([x.size(0) > 0 for x in new_batch['kg_adj']]), print(batch)

        if save:
            new_batch['info'] = np.stack([np.array(np.pad(b['info'],
                                                          ((0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0]),
                                                           (0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0])),
                                                          'constant',
                                                          constant_values=(-1, -1)), dtype='object') for b in batch], axis=0)

        return new_batch
