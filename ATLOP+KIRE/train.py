import argparse
import os

import numpy as np
import torch
# from apex import amp
import time
import json
# import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
# import wandb
from knowledge_injection_layer.kg_data_loader import KG_data_loader
from knowledge_injection_layer.config import Config as kgConfig
import yaml
import yamlordereddictloader

def train(args, model, train_features, dev_features, test_features):
    train_kg_loader = KG_data_loader("dev_train", dataset=args.dataset)
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))

        init_add_kg_flag = kgConfig.add_kg_flag
        init_add_coref_flag = kgConfig.add_coref_flag
        if kgConfig.train_method in ['two_step']:
            kgConfig.add_kg_flag = False
            kgConfig.add_coref_flag = False
            train_step = 1 # reload basic RE model
        else:
            optimizer.param_groups[0]['lr'] = 1e-3
            optimizer.param_groups[3]['lr'] = 1e-3
            train_step = 2
        cur_patience = 0
        for epoch in train_iterator:
            t1 = time.time()
            if kgConfig.train_method == 'two_step' and train_step == 1:
                print("load pretrained basic RE model")
                
                miss_keys, unexpect_keys = model.load_state_dict(torch.load('results_bert/re.model'), strict=False)
                
                print("miss_keys", miss_keys)
                print("unexpect_keys", unexpect_keys)
                # model.load_state_dict(torch.load(args.save_path))
                if init_add_kg_flag:
                    kgConfig.add_kg_flag = True
                if init_add_coref_flag:
                    kgConfig.add_coref_flag = True
                optimizer.param_groups[0]['lr'] = kgConfig.kg_lr
                optimizer.param_groups[1]['lr'] = kgConfig.other_lr
                optimizer.param_groups[3]['lr'] = kgConfig.kg_lr
                optimizer.param_groups[4]['lr'] = kgConfig.other_lr
                train_step = 2
                best_score = 0

            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                dids = batch[5]
                input_lengths = (batch[0] > 0).long().sum(dim=1)
                batch_max_length1 = int(input_lengths.max())
                # print(batch_max_length1)
                batch_max_length = max([len(x) for x in batch[6]])
                # print(batch_max_length1, batch_max_length)

                train_kg_batch = train_kg_loader.get_kg_batch(dids, batch_max_length)
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'subword_indexs': batch[6],
                          'kg_ent_mask': train_kg_batch['kg_ent_mask'],
                          'kg_ent_attrs': train_kg_batch['kg_ent_attrs'],
                          'kg_ent_attr_lens': train_kg_batch['kg_ent_attr_lens'],
                          'kg_ent_attr_nums': train_kg_batch['kg_ent_attr_nums'],
                          'kg_ent_labels': train_kg_batch['kg_ent_labels'],
                          'kg_adj': train_kg_batch['kg_adj'], 'kg_adj_edges': train_kg_batch['kg_adj_edges'],
                          'coref_h_mapping': train_kg_batch['coref_h_mapping'],
                          'coref_t_mapping': train_kg_batch['coref_t_mapping'],
                          'coref_dis': train_kg_batch['coref_dis'],
                          'coref_lens': train_kg_batch['coref_lens'],
                          'coref_label': train_kg_batch['coref_label'],
                          'coref_label_mask': train_kg_batch['coref_label_mask'],
                          'coref_mention_position': train_kg_batch['coref_mention_position'],
                          }
                outputs, loss_kg, loss_coref = model(**inputs)
                loss_re = outputs[0] / args.gradient_accumulation_steps

                loss = model.combineloss(loss_re, loss_coref, loss_kg)

                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # wandb.log({"loss": loss.item()}, step=num_steps)
            t2 = time.time()
            print("epoch time", t2-t1)
            # if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
            if True:
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    # wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        # pred = report(args, model, test_features)
                        # with open("result.json", "w") as fh:
                        #     json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                        cur_patience = 0
                    else: #if (step + 1) == len(train_dataloader) - 1:
                        cur_patience += 1
            if cur_patience > 15:
                if train_step==0:
                    train_step=1
                    cur_patience=0
                else:
                    print(best_score)
                    print("early stop")
                    break
        return num_steps

    new_layer = ["extractor", "bilinear"]

    paramsbert = []  # lr=0 or 1e-5
    paramsbert0reg = []
    for p_n, p in model.model.named_parameters():
        if not p.requires_grad:
            continue
        if '.bias' in p_n:
            paramsbert0reg.append(p)
        else:
            paramsbert.append(p)
    paramsbert_ids = list(map(id, paramsbert)) + list(map(id, paramsbert0reg))

    if kgConfig.add_kg_flag:
        paramskg = [param for p_name, param in model.kg_injection.named_parameters() if
                    param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  # step 2
        paramskg0reg = [param for p_name, param in model.kg_injection.named_parameters() if
                        param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  # step 2

        paramskg_ids = list(map(id, paramskg)) + list(map(id, paramskg0reg))
    else:
        paramskg = []
        paramskg0reg = []
        paramskg_ids = []

    if kgConfig.add_coref_flag:
        paramskg += [param for p_name, param in model.coref_injection.named_parameters() if
                     param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  # step 2
        paramskg0reg += [param for p_name, param in model.coref_injection.named_parameters() if
                         param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  # step 2
        paramskg_ids = list(map(id, paramskg)) + list(map(id, paramskg0reg))

    paramsothers = [param for p_name, param in model.named_parameters() if param.requires_grad and (
            id(param) not in (paramskg_ids + paramsbert_ids)) and '.bias' not in p_name]
    paramsothers0reg = [param for p_name, param in model.named_parameters() if param.requires_grad and (
            id(param) not in (paramskg_ids + paramsbert_ids)) and '.bias' in p_name]
    groups = [dict(params=paramskg, lr=0.0), dict(params=paramsothers, lr=1e-4),
              dict(params=paramsbert, lr=args.learning_rate),
              dict(params=paramskg0reg, lr=0.0, weight_decay=0.0),
              dict(params=paramsothers0reg, lr=1e-4, weight_decay=0.0),
              dict(params=paramsbert0reg, lr=args.learning_rate, weight_decay=0.0)]

    optimizer = AdamW(groups, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    if tag=='dev':
        dev_kg_loader = KG_data_loader("dev_dev", dataset=args.dataset)
    elif tag=='test':
        dev_kg_loader = KG_data_loader("dev_test", dataset=args.dataset)
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        dids = batch[5]
        input_lengths = (batch[0] > 0).long().sum(dim=1)
        batch_max_length = int(input_lengths.max())
        # print(batch_max_length)
        batch_max_length = max([len(x) for x in batch[6]])
        # print(batch_max_length)
        dev_kg_batch = dev_kg_loader.get_kg_batch(dids, batch_max_length)
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'subword_indexs': batch[6],
                  'kg_ent_mask': dev_kg_batch['kg_ent_mask'],
                  'kg_ent_attrs': dev_kg_batch['kg_ent_attrs'],
                  'kg_ent_attr_lens': dev_kg_batch['kg_ent_attr_lens'],
                  'kg_ent_attr_nums': dev_kg_batch['kg_ent_attr_nums'],
                  'kg_ent_labels': dev_kg_batch['kg_ent_labels'],
                  'kg_adj': dev_kg_batch['kg_adj'], 'kg_adj_edges': dev_kg_batch['kg_adj_edges'],
                  'coref_h_mapping': dev_kg_batch['coref_h_mapping'],
                  'coref_t_mapping': dev_kg_batch['coref_t_mapping'],
                  'coref_dis': dev_kg_batch['coref_dis'],
                  'coref_lens': dev_kg_batch['coref_lens'],
                  'coref_label': dev_kg_batch['coref_label'],
                  'coref_label_mask': dev_kg_batch['coref_label_mask'],
                  'coref_mention_position': dev_kg_batch['coref_mention_position']
                  }

        with torch.no_grad():
            pred, _, _ = model(**inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features, args.dataset)
    # print("ans", ans[:10])
    if len(ans) > 0:
        best_f1, p, r,  _, best_f1_ign, p_ign, r_ign, _ = official_evaluate(ans, args.data_dir, tag)
    else:
        best_f1, p, r, best_f1_ign, p_ign, r_ign = 0, 0, 0, 0, 0, 0
    output = {
        tag + "_P": p*100,
        tag + "_R": r*100,
        tag + "_F1": best_f1 * 100,
        tag + "_P_ign": p_ign * 100,
        tag + "_R_ign": r_ign * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }

    return best_f1, output

def report(args, model, features, tag="test"):
    if tag=='dev':
        dev_kg_loader = KG_data_loader("dev_dev", dataset=args.dataset)
    elif tag=='test':
        dev_kg_loader = KG_data_loader("dev_test", dataset=args.dataset)
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()
        dids = batch[5]
        input_lengths = (batch[0] > 0).long().sum(dim=1)
        batch_max_length = int(input_lengths.max())
        batch_max_length = max([len(x) for x in batch[6]])
        dev_kg_batch = dev_kg_loader.get_kg_batch(dids, batch_max_length)
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'subword_indexs': batch[6],
                  'kg_ent_mask': dev_kg_batch['kg_ent_mask'],
                  'kg_ent_attrs': dev_kg_batch['kg_ent_attrs'],
                  'kg_ent_attr_lens': dev_kg_batch['kg_ent_attr_lens'],
                  'kg_ent_attr_nums': dev_kg_batch['kg_ent_attr_nums'],
                  'kg_ent_labels': dev_kg_batch['kg_ent_labels'],
                  'kg_adj': dev_kg_batch['kg_adj'], 'kg_adj_edges': dev_kg_batch['kg_adj_edges'],
                  'coref_h_mapping': dev_kg_batch['coref_h_mapping'],
                  'coref_t_mapping': dev_kg_batch['coref_t_mapping'],
                  'coref_dis': dev_kg_batch['coref_dis'],
                  'coref_lens': dev_kg_batch['coref_lens'],
                  'coref_label': dev_kg_batch['coref_label'],
                  'coref_label_mask': dev_kg_batch['coref_label_mask'],
                  'coref_mention_position': dev_kg_batch['coref_mention_position']
                  }

        with torch.no_grad():
            pred, _, _ = model(**inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features, args.dataset)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="docred", type=str)
    parser.add_argument("--data_dir", default="./data/DocRED", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument('--kgfile', type=str, default='None')

    args = parser.parse_args()
    if args.kgfile != 'None':
        with open(args.kgfile, 'r', encoding="utf-8") as f:
            kg_params = yaml.load(f, Loader=yamlordereddictloader.Loader)
            # print(kg_params)
            for k, v in kg_params.items():
                setattr(kgConfig, k, v)

    # wandb.init(project="DocRED")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,  output_attentions=True,
        num_labels=args.num_class,
    )
    config.output_attentions = True
    config.num_labels = args.num_class

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length, dataset=args.dataset)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, dataset=args.dataset)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, dataset=args.dataset)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    assert model.config.output_attentions==True,print(model.config.output_attentions)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path), strict=False)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open(os.path.join(args.load_path[:-8], "result.json"), "w") as fh:
            json.dump(pred, fh)
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)


if __name__ == "__main__":
    main()
