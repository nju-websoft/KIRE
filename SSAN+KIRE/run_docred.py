import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from dataset import docred_convert_examples_to_features as convert_examples_to_features
from dataset import DocREDProcessor

from model import (BertForDocRED, RobertaForDocRED)
logger = logging.getLogger(__name__)

# Modify code here
from knowledge_injection_layer.kg_data_loader import KG_data_loader
from knowledge_injection_layer.config import Config as kgConfig
import yaml
import yamlordereddictloader
from torch.nn.utils.rnn import pad_sequence

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_kg_loader = KG_data_loader("dev_train", dataset=args.dataset)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.warmup_ratio > 0:
        args.warmup_steps = t_total * args.warmup_ratio

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    # set grad: train all params
    for name, param in model.named_parameters():
        param.requires_grad = True

    # Modify code here:
    paramsbert = []  
    paramsbert0reg = []
    for p_n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if kgConfig.add_kg_flag and 'kg' in p_n:
            continue
        if kgConfig.add_coref_flag and 'coref' in p_n:
            continue
        if ('bias' in p_n) or ('LayerNorm.weight' in p_n):
            paramsbert0reg.append(p)
        else:
            paramsbert.append(p)
    paramsbert_ids = list(map(id, paramsbert)) + list(map(id, paramsbert0reg))
    lr_kg = 0.0
    lr_other = 0.0
    count = 0
    if kgConfig.add_kg_flag:
        lr_kg = kgConfig.kg_lr
        lr_other = kgConfig.other_lr
        paramskg = [param for p_name, param in model.kg_injection.named_parameters() if
                    param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  
        paramskg0reg = [param for p_name, param in model.kg_injection.named_parameters() if
                        param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  
    else:
        paramskg = []
        paramskg0reg = []

    if kgConfig.add_coref_flag:
        lr_kg = kgConfig.kg_lr
        lr_other = kgConfig.other_lr
        paramskg += [param for p_name, param in model.coref_injection.named_parameters() if
                     param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' not in p_name]  
        paramskg0reg += [param for p_name, param in model.coref_injection.named_parameters() if
                         param.requires_grad and (id(param) not in paramsbert_ids) and '.bias' in p_name]  
    #paramsbert =  [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    #paramsbert0reg = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    lr_kg = float(lr_kg)
    lr_other = float(lr_other)
    logging.info(lr_kg)
    logging.info(lr_other)
    groups = [dict(params=paramskg, lr=lr_kg), 
              dict(params=paramsbert, weight_decay = args.weight_decay),
              dict(params=paramskg0reg, lr=lr_other, weight_decay=0.0),
              dict(params=paramsbert0reg, weight_decay=0.0)]

    '''
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    '''
    optimizer = AdamW(groups, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.lr_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, bert_tr_loss, bert_logging_loss = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    # Modify code here:
    init_add_kg_flag = kgConfig.add_kg_flag
    init_add_coref_flag = kgConfig.add_coref_flag
    if kgConfig.train_method in ['two_step']:
        kgConfig.add_kg_flag = False
        kgConfig.add_coref_flag = False
        train_step = 1 # reload basic RE model
    else:
        print("happy new year")
        #optimizer.param_groups[0]['lr'] = 1e-3
        #optimizer.param_groups[2]['lr'] = 1e-3
        train_step = 2

    best_f1 = 0
    for _ in train_iterator:
        if kgConfig.train_method == 'two_step' and train_step == 1:
            print("load pretrained basic RE model")
            
                #miss_keys, unexpect_keys = model.load_state_dict(torch.load('results_bert/re.model'), strict=False)
            miss_keys, unexpect_keys = model.load_state_dict(torch.load('./remodel/pytorch_model.bin'), strict=False)
            print("miss_keys", miss_keys)
            print("unexpect_keys", unexpect_keys)
            
            if init_add_kg_flag:
                kgConfig.add_kg_flag = True
            if init_add_coref_flag:
                kgConfig.add_coref_flag = True
            optimizer.param_groups[0]['lr'] = kgConfig.kg_lr
            optimizer.param_groups[2]['lr'] = kgConfig.kg_lr
            print(optimizer.param_groups[0]['lr'])
            print(optimizer.param_groups[2]['lr'])
            train_step = 2

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            dids = batch[11]
            input_lengths = (batch[0] > 0).long().sum(dim=1)
            batch_max_length1 = int(input_lengths.max())
            # print(batch_max_length1)
            #batch_max_length = max([len(x) for x in batch[10]])
            # print(batch_max_length1, batch_max_length)\
            if args.dataset == "DocRED":
                batch_max_length = 512
            else:
                batch_max_length = 1800
            train_kg_batch = train_kg_loader.get_kg_batch(dids, batch_max_length)
            
            # roberta does not accept token_type_ids
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "subword_indexs": batch[10],
                      'kg_ent_mask': train_kg_batch['kg_ent_mask'],
                      'kg_ent_attrs': train_kg_batch['kg_ent_attrs'],
                      'kg_ent_attr_lens': train_kg_batch['kg_ent_attr_lens'],
                      'kg_ent_attr_nums': train_kg_batch['kg_ent_attr_nums'],
                      'kg_ent_attr_indexs': train_kg_batch['kg_ent_attr_indexs'],
                      'kg_ent_labels': train_kg_batch['kg_ent_labels'].long(),
                      'kg_adj': train_kg_batch['kg_adj'], 'kg_adj_edges': train_kg_batch['kg_adj_edges'],
                      'kg_radj': train_kg_batch['kg_radj'],
                      'coref_h_mapping': train_kg_batch['coref_h_mapping'],
                      'coref_t_mapping': train_kg_batch['coref_t_mapping'],
                      'coref_dis': train_kg_batch['coref_dis'],
                      'coref_lens': train_kg_batch['coref_lens'],
                      'coref_label': train_kg_batch['coref_label'],
                      'coref_label_mask': train_kg_batch['coref_label_mask'],
                      'coref_mention_position': train_kg_batch['coref_mention_position'],
                      }

            outputs, loss_kg, loss_coref = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            total = sum([param.nelement() for param in model.parameters()])
            print("number is " +str(total))
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            bert_re_loss = loss
            loss = model.combineloss(loss, loss_coref, loss_kg)
            loss.backward()

            bert_tr_loss += bert_re_loss.item()
            tr_loss +=loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results,_,_ = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    bert_loss_scalar = (bert_tr_loss - bert_logging_loss) / args.logging_steps
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logger.info(scheduler.get_lr())
                    learning_rate_scalar = scheduler.get_lr()[1]
                    lr = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss
                    bert_logging_loss = bert_tr_loss
                    logger.info("bert_learning_rate:{0}, kg_lr:{1}, loss_bert:{2}, loss_all:{3}, step:{4}   ".format(learning_rate_scalar, lr, bert_loss_scalar, loss_scalar, global_step))
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        _, dev_f1, output_preds_thresh = evaluate(args, model, tokenizer)
        if(dev_f1>best_f1):
            best_f1 = dev_f1
            count = 0
            if best_f1 > 0.4:
                if args.output_dir is None and args.local_rank in [-1, 0]:
                    raise ValueError('checkpoint_dir is not set!')
                if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                    os.makedirs(args.output_dir)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                # write pred file
                output_eval_file = os.path.join(args.checkpoint_dir, "eval_result.json")
                with open(output_eval_file, 'w') as f:
                    json.dump(output_preds_thresh, f)
        else:
            count=count+1
        if count > 15:
            break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Modify code here:
    dev_kg_loader = KG_data_loader("dev_dev", dataset=args.dataset)

    processor = DocREDProcessor(args.dataset)
    dev_examples = processor.get_dev_examples(args.data_dir)

    label_map = processor.get_label_map(args.data_dir)
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        input_lengths = (batch[0] > 0).long().sum(dim=1)
        batch_max_length = int(input_lengths.max())
        if args.dataset == "DocRED":
                batch_max_length = 512
        else:
            batch_max_length = 1800
        dids = batch[11]
        dev_kg_batch = dev_kg_loader.get_kg_batch(dids, batch_max_length)
      
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "subword_indexs": batch[10],
                      'kg_ent_mask': dev_kg_batch['kg_ent_mask'],
                      'kg_ent_attrs': dev_kg_batch['kg_ent_attrs'],
                      'kg_ent_attr_lens': dev_kg_batch['kg_ent_attr_lens'],
                      'kg_ent_attr_nums': dev_kg_batch['kg_ent_attr_nums'],
                      'kg_ent_attr_indexs': dev_kg_batch['kg_ent_attr_indexs'],
                      'kg_ent_labels': dev_kg_batch['kg_ent_labels'],
                      'kg_adj': dev_kg_batch['kg_adj'], 'kg_adj_edges': dev_kg_batch['kg_adj_edges'],
                      'kg_radj': dev_kg_batch['kg_radj'],
                      'coref_h_mapping': dev_kg_batch['coref_h_mapping'],
                      'coref_t_mapping': dev_kg_batch['coref_t_mapping'],
                      'coref_dis': dev_kg_batch['coref_dis'],
                      'coref_lens': dev_kg_batch['coref_lens'],
                      'coref_label': dev_kg_batch['coref_label'],
                      'coref_label_mask': dev_kg_batch['coref_label_mask'],
                      'coref_mention_position': dev_kg_batch['coref_mention_position']
                      }

            outputs,_,_ = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = inputs["ent_mask"].detach().cpu().numpy()
            out_label_ids = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, inputs["ent_mask"].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    total_labels = 0
    output_preds = []
    for (i, (example, pred, ent_mask)) in enumerate(zip(dev_examples, preds, ent_masks)):

        spo_gt_tmp = []
        for spo_gt in example.labels:
            spo_gt_tmp.append((spo_gt['h'], spo_gt['t'], spo_gt['r']))
        total_labels += len(spo_gt_tmp)

        for h in range(len(example.vertexSet)):
            for t in range(len(example.vertexSet)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit in enumerate(pred[h][t]):
                    if predicate_id == 0:
                        continue
                    if (h, t, predicate_map[predicate_id]) in spo_gt_tmp:
                        flag = True
                    else:
                        flag = False
                    output_preds.append((flag, logit, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key = lambda x: x[1], reverse=True)
    pr_x = []
    pr_y = []
    correct = 0
    for i, pred in enumerate(output_preds):
        correct += pred[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / total_labels)

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = output_preds[f1_pos][1]

    output_preds_thresh = []
    for i in range(f1_pos+1):
        output_preds_thresh.append({"title": output_preds[i][2],
                                    "h_idx": output_preds[i][3],
                                    "t_idx": output_preds[i][4],
                                    "r": output_preds[i][5],
                                    "evidence": []
                                    })

    result = {"loss": eval_loss, "precision": pr_y[f1_pos],
              "recall": pr_x[f1_pos], "f1": f1, "thresh": thresh}
    logger.info(result)
    


    
    return result, f1, output_preds_thresh


def predict(args, model, tokenizer, prefix=""):
    # Modify code here:
    dev_kg_loader = KG_data_loader("dev_test", dataset=args.dataset)

    processor = DocREDProcessor(args.dataset)
    pred_examples = processor.get_test_examples(args.data_dir)

    label_map = processor.get_label_map(args.data_dir)
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    eval_dataset = load_and_cache_examples(args, tokenizer, predict=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        input_lengths = (batch[0] > 0).long().sum(dim=1)
        batch_max_length = int(input_lengths.max())
        # print(batch_max_length)
        #batch_max_length = max([len(x) for x in batch[10]])
        # print(batch_max_length)
        # Modify code here:
        if args.dataset == "DocRED":
            batch_max_length = 512
        else:
            batch_max_length = 1800
        dids = batch[11]
        dev_kg_batch = dev_kg_loader.get_kg_batch(dids, batch_max_length)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2] if args.model_type == 'bert' else None,
                      "ent_mask": batch[3],
                      "ent_ner": batch[4],
                      "ent_pos": batch[5],
                      "ent_distance": batch[6],
                      "structure_mask": batch[7],
                      "label": batch[8],
                      "label_mask": batch[9],
                      "subword_indexs": batch[10],
                      'kg_ent_mask': dev_kg_batch['kg_ent_mask'],
                      'kg_ent_attrs': dev_kg_batch['kg_ent_attrs'],
                      'kg_ent_attr_lens': dev_kg_batch['kg_ent_attr_lens'],
                      'kg_ent_attr_nums': dev_kg_batch['kg_ent_attr_nums'],
                      'kg_ent_attr_indexs': dev_kg_batch['kg_ent_attr_indexs'],
                      'kg_ent_labels': dev_kg_batch['kg_ent_labels'],
                      'kg_adj': dev_kg_batch['kg_adj'], 'kg_adj_edges': dev_kg_batch['kg_adj_edges'],
                      'kg_radj': dev_kg_batch['kg_radj'],
                      'coref_h_mapping': dev_kg_batch['coref_h_mapping'],
                      'coref_t_mapping': dev_kg_batch['coref_t_mapping'],
                      'coref_dis': dev_kg_batch['coref_dis'],
                      'coref_lens': dev_kg_batch['coref_lens'],
                      'coref_label': dev_kg_batch['coref_label'],
                      'coref_label_mask': dev_kg_batch['coref_label_mask'],
                      'coref_mention_position': dev_kg_batch['coref_mention_position']
                      }

            outputs,_,_ = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = inputs["ent_mask"].detach().cpu().numpy()
            out_label_ids = inputs["label"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, inputs["ent_mask"].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    print("eval_loss: {}".format(eval_loss))
    if args.dataset == "DocRED":
        output_preds = []
        for (i, (example, pred, ent_mask)) in enumerate(zip(pred_examples, preds, ent_masks)):
            for h in range(len(example.vertexSet)):
                for t in range(len(example.vertexSet)):
                    if h == t:
                        continue
                    if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                        continue
                    for predicate_id, logit in enumerate(pred[h][t]):
                        if predicate_id == 0:
                            continue
                        else:
                            output_preds.append((logit, example.title, h, t, predicate_map[predicate_id]))
        output_preds.sort(key=lambda x: x[0], reverse=True)
        output_preds_thresh = []
        for i in range(len(output_preds)):
            if output_preds[i][0] < args.predict_thresh:
                break
            output_preds_thresh.append({"title": output_preds[i][1],
                                        "h_idx": output_preds[i][2],
                                        "t_idx": output_preds[i][3],
                                        "r": output_preds[i][4],
                                        "evidence": []
                                        })
        # write pred file
        if not os.path.exists('./data/DocRED/') and args.local_rank in [-1, 0]:
            os.makedirs('./data/DocRED')
        output_eval_file = os.path.join(args.checkpoint_dir, "test_result.json")
        with open(output_eval_file, 'w') as f:
            json.dump(output_preds_thresh, f)
    elif args.dataset == "DWIE":
        total_labels = 0
        output_preds = []
        for (i, (example, pred, ent_mask)) in enumerate(zip(pred_examples, preds, ent_masks)):

            spo_gt_tmp = []
            for spo_gt in example.labels:
                spo_gt_tmp.append((spo_gt['h'], spo_gt['t'], spo_gt['r']))
            total_labels += len(spo_gt_tmp)

            for h in range(len(example.vertexSet)):
                for t in range(len(example.vertexSet)):
                    if h == t:
                        continue
                    if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                        continue
                    for predicate_id, logit in enumerate(pred[h][t]):
                        if predicate_id == 0:
                            continue
                        if (h, t, predicate_map[predicate_id]) in spo_gt_tmp:
                            flag = True
                        else:
                            flag = False
                        output_preds.append((flag, logit, example.title, h, t, predicate_map[predicate_id]))
        output_preds.sort(key = lambda x: x[1], reverse=True)
        pr_x = []
        pr_y = []
        correct = 0
        for i, pred in enumerate(output_preds):
            correct += pred[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_labels)

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        thresh = output_preds[f1_pos][1]

        output_preds_thresh = []
        for i in range(f1_pos+1):
            output_preds_thresh.append({"title": output_preds[i][2],
                                        "h_idx": output_preds[i][3],
                                        "t_idx": output_preds[i][4],
                                        "r": output_preds[i][5],
                                        "evidence": []
                                        })

        result = {"loss": eval_loss, "precision": pr_y[f1_pos],
                "recall": pr_x[f1_pos], "f1": f1, "thresh": thresh}
        logger.info(result)
        output_eval_file = os.path.join(args.checkpoint_dir, "test_result.json")
        with open(output_eval_file, 'w') as f:
            json.dump(output_preds_thresh, f)

def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = DocREDProcessor(args.dataset)
    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_map = processor.get_label_map(args.data_dir)

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    elif predict:
        examples = processor.get_test_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    features = convert_examples_to_features(
        examples,
        args.model_type,
        tokenizer,
        args.dataset,
        max_length=args.max_seq_length,
        max_ent_cnt=args.max_ent_cnt,
        label_map=label_map
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    '''
    subword_indexs, dids, ep_dis = [],[],[]
    for f in features:
        subword_indexs.append(f.subword_indexs)
        dids.append(f.dids)
        ep_dis.append(f.ep_dis)
    '''
    

    subword_indexs=pad_sequence([torch.tensor(f.subword_indexs,dtype=torch.long) for f in features], batch_first = True)
    #dids = pad_sequence([torch.tensor(f.dids, dtype=torch.long) for f in features], batch_first = True)
    dids = torch.tensor([f.dids for f in features], dtype=torch.long)
    #ep_dis = pad_sequence([torch.tensor(f.ep_dis, dtype=torch.long) for f in features], batch_first = True)
    #print(ep_dis.shape())
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_attention_mask = torch.tensor(np.array([f.attention_mask for f in features]), dtype=torch.long)
    all_token_type_ids = torch.tensor(np.array([f.token_type_ids for f in features]), dtype=torch.long)
    all_ent_mask = torch.tensor(np.array([f.ent_mask for f in features]), dtype=torch.float)
    all_ent_ner = torch.tensor(np.array([f.ent_ner for f in features]), dtype=torch.long)
    all_ent_pos = torch.tensor(np.array([f.ent_pos for f in features]), dtype=torch.long)
    all_ent_distance = torch.tensor(np.array([f.ent_distance for f in features]), dtype=torch.long)
    all_structure_mask = torch.tensor(np.array([f.structure_mask for f in features]), dtype=torch.bool)
    all_label = torch.tensor(np.array([f.label for f in features]), dtype=torch.bool)
    all_label_mask = torch.tensor(np.array([f.label_mask for f in features]), dtype=torch.bool)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance,
                            all_structure_mask, all_label, all_label_mask, subword_indexs, dids)

    return dataset


def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    #torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    # Modify code here:
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )
    parser.add_argument("--no_naive_feature", action="store_true",
                        help="do not exploit naive features for DocRED, include ner tag, entity id, and entity pair distance")
    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.5, type=float, help="pred thresh")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--lr_schedule", default='linear', type=str, choices=['linear', 'constant'],
                        help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument('--kgfile', type=str, default='None')
    args = parser.parse_args()
    
    # Modify code here:
    if args.kgfile != 'None':
        with open(args.kgfile, 'r', encoding="utf-8") as f:
            kg_params = yaml.load(f, Loader=yamlordereddictloader.Loader)
            # print(kg_params)
            for k, v in kg_params.items():
                setattr(kgConfig, k, v)
    
    ModelArch = None
    if args.model_type == 'roberta':
        ModelArch = RobertaForDocRED
    elif args.model_type == 'bert':
        ModelArch = BertForDocRED
    if args.no_naive_feature:
        with_naive_feature = False
    else:
        with_naive_feature = True
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    if args.do_train:
            args.checkpoint_dir = args.output_dir
    elif args.checkpoint_dir is None and args.local_rank in [-1, 0]:
        raise ValueError('checkpoint_dir is not set!')
    # Set seed
    set_seed(args)
    processor = DocREDProcessor(args.dataset)
    label_map = processor.get_label_map(args.data_dir)
    num_labels = len(label_map.keys())

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        # Modify code here:
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        model = ModelArch.from_pretrained(args.model_name_or_path,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          dataset = args.dataset,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)
        logger.info("Training parameters %s", args)
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        '''
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if args.output_dir is None and args.local_rank in [-1, 0]:
                raise ValueError('checkpoint_dir is not set!')
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        '''
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            args.checkpoint_dir = args.output_dir
        elif args.checkpoint_dir is None and args.local_rank in [-1, 0]:
            raise ValueError('checkpoint_dir is not set!')
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        # Modify code here:
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          dataset = args.dataset,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        result, _, _= evaluate(args, model, tokenizer)
        results.update(result)

    # predict
    if args.do_predict and args.local_rank in [-1, 0]:
        # Modify code here:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, do_lower_case=args.do_lower_case)
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        model = ModelArch.from_pretrained(args.checkpoint_dir,
                                          from_tf=bool(".ckpt" in args.model_name_or_path),
                                          config=config,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          num_labels=num_labels,
                                          max_ent_cnt=args.max_ent_cnt,
                                          dataset = args.dataset,
                                          with_naive_feature=with_naive_feature,
                                          entity_structure=args.entity_structure,
                                          )
        model.to(args.device)
        predict(args, model, tokenizer)

    return results

if __name__ == "__main__":
    main()
