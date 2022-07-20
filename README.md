# Enhancing Document-level Relation Extraction by Entity Knowledge Injection
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/KIRE/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/KIRE/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-Pytorch-orange.svg?style=flat-square)](https://pytorch.org/)

> Document-level relation extraction (RE) aims to identify the relations between entities throughout an entire document. It needs complex reasoning skills to synthesize various knowledge such as coreferences and commonsense. Large-scale knowledge graphs (KGs) contain a wealth of real-world facts, and can provide valuable knowledge to document-level RE. In this paper, we propose an entity knowledge injection framework to enhance current document-level RE models. Specifically, we introduce coreference distillation to inject coreference knowledge, endowing an RE model with the more general capability of coreference reasoning. We also employ representation reconciliation to inject factual knowledge and aggregate KG representations and document representations into a unified space. The experiments on two benchmark datasets validate the generalization of our entity knowledge injection framework and the consistent improvement to several document-level RE models.

## Table of contents📑

1. [Introduction of KIRE](#introduction-of-KIRE)
   1. [Overview](#overview)
   2. [Package Description](#package-description)
2. [Getting Started](#getting-started)
   1. [Dependencies](#dependencies)
   2. [Downloads](#downloads)
   3. [Usage](#usage)
3. [Models](#models)
   1. [Document-level RE models](#re-models)
   2. [Knowledge injection models](#ki-models)
4. [Datasets](#datasets)
   1. [DocRED dataset](#docred-dataset)
   2. [DWIE dataset](#dwie-dataset)
5. [License](#license)
6. [Citation](#citation)

## Introduction of KIRE🎖️
### Overview

We use  [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) to develop the basic framework of **KIRE**.  The framework architecture is illustrated in the following Figure. 

![image](https://github.com/nju-websoft/KIRE/blob/main/figs/model.png)

### 	Package Description

```python
KIRE/
├── B4+KIRE/
│   ├── configs/  # the code for running the model
│   ├── checkpoints/  # store the model after train
│   ├── prepro_data/  # the preprocessed data
│   ├── model/  # the models include CNN, LSTM, BiLSTM, Context-aware
│   ├── knowledge_injection_layer/  # the knowledge injection module
│   ├── scripts/  # different code files corresponding to the sh files in the home directory
├── GLRE+KIRE/
│   ├── configs/  # different configs used for experiments
│   ├── data/  # datasets and corresponding data loading code
│   ├── data_processing/  # the preprocess code for datasets
│   ├── knowledge_injection_layer/  # the knowledge injection module
│   ├── scripts/  #  different code files corresponding to the sh files in the home directory
│   ├── other directories contain source code of GLRE model
├── SSAN+KIRE/
│   ├── data/  # datasets and corresponding generate code
│   ├── checkpoints/  # store the model after train
│   ├── pretrained_lm/  # store the pretrained model
│   ├── knowledge_injection_layer/  # the knowledge injection module
│   ├── other directories or files contain source code of SSAN model
├── ATLOP+KIRE/
│   ├── data/  # datasets and corresponding generate code
│   ├── knowledge_injection_layer/  # the knowledge injection module
│   ├── scripts/  # different sh files used for experiments under different settings
│   ├── other directories or files contain source code of ATLOP model
```

## Getting Started✈️

### Dependencies

* Python (tested on 3.7.4)
* CUDA 
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 2.11.0)
* numpy
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm
* yamlordereddictloader (tested on 0.4.0)
* scipy (tested on 1.5.2)
* recordtype
* tabulate
* scikit-learn

### Downloads
* Download processed data from [figshare](https://figshare.com/articles/dataset/Processed_data/14602203) 
* Download pretrained autoencoder model from [figshare](https://figshare.com/articles/dataset/re_model/14602185) 
* Download pretrained language model from [huggingface](https://huggingface.co/bert-base-uncased) 

### Usage

To run the off-the-shelf approaches and reproduce our experiments, we choose ATLOP model as an example.
#### Train on DocRED
Train the ATLOP and KIRE + ATLOP model on DocRED with the following command:

```bash
>> sh scripts/run_docred_bert.sh  # for ATLOP_BERT_base model
>> sh scripts/run_docred_bert_kire.sh  # for ATLOP_BERT_base + KIRE model 
```

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

#### Train on DWIE

Train the ATLOP and KIRE + ATLOP model on DWIE with the following command:

```bash
>> sh scripts/run_dwie_bert.sh  # for ATLOP_BERT_base model
>> sh scripts/run_dwie_bert_kire.sh  # for ATLOP_BERT_base + KIRE model 
```

The scripts to run other basic models with the KIRE framework can be found in their corresponding directories.

The following table shows the used hyperparameter values in the experiments.

| Hyperparameter    | Values                                                                                                                |
| -------- |-------------------------------------------------------------------------------------------------------------------------|
| Batch size   | 4 |
| Learning rate   |  0.0005 |
| Gradient clipping   |  10 |
| Early stop patience   | 10 |
| Regularization   | 0.0001 |
| Dropout ratio  | 0.2 or 0.5 |
| Dimension of hidden layers in MLP   | 256 |
| Dimension of GloVe and Skip-gram   | 100 |
| Dimension of hidden layers in AutoEncoder   |  50 |
| Dimension, kernel size and stride of CNN_{1D}    |  100,3,1 |
| Number of R-GAT layers and heads   | 3, 2 |
| Number of aggregators   | 2 |
| Dimension of hidden layers in aggregation  | 768 |
| 𝛼1, 𝛼2, 𝛼3    | 1, 0.01, 0.01 |

## Models🤖

### Document-level RE models
KIRE utilizes 7 basic document-level relation extraction models. The citation for each models corresponds to either the paper describing the model.

| Name     | Citation                                                                                                                |
| -------- |-------------------------------------------------------------------------------------------------------------------------|
| CNN   | [Yao *et al.*, 2019](https://arxiv.org/abs/1906.06127v3) |
| LSTM   |  [Yao *et al.*, 2019](https://arxiv.org/abs/1906.06127v3) |
| BiLSTM   |  [Yao *et al.*, 2019](https://arxiv.org/abs/1906.06127v3) |
| Context-aware   | [Yao *et al.*, 2019](https://arxiv.org/abs/1906.06127v3) |
| GLRE   | [Wang *et al.*, 2020](https://aclanthology.org/2020.emnlp-main.303.pdf)                                                            |
| SSAN   | [Xu *et al.*, 2020](https://arxiv.org/abs/2102.10249)                                                                |
| ATLOP   | [Zhou *et al.*, 2020](https://arxiv.org/abs/2010.11304)     |

### Knowledge injection models
KIRE chooses 3 basic knowledge injection models as competitors. The citation for each models corresponds to either the paper describing the model.

| Name     | Citation                                                                                                                |
| -------- |-------------------------------------------------------------------------------------------------------------------------|
| RESIDE   | [Vashishth *et al.*, 2018](https://arxiv.org/abs/1812.04361) |
| RECON   |  [Bastos *et al.*, 2019](https://dl.acm.org/doi/abs/10.1145/3442381.3449917) |
| KB-graph   |  [Verlinden *et al.*, 2021](https://arxiv.org/abs/2107.02286) |


## Datasets🗂️

KIRE selects two benchmark document-level relation extraction datasets: [DocRED](https://github.com/thunlp/DocRED) and [DWIE](https://github.com/klimzaporojets/DWIE).
The statistical data is listed in the following tables.

### DocRED dataset
| Datasets | Documents | Relation types | Instances |N/A instances|
|:-----:|:------------:|:---------:|:--------:|:--------:|
|        Training        |    3,053     |     96      |  38,269  |  1,163,035  |
|        Validation        |    1,000     |     96     |  12,332  |  385,263  |
|        Test        |    1,000     |     96      |  12,842  |  379,316  |

### DWIE dataset
| Datasets | Documents | Relation types | Instances |N/A instances|
|:-----:|:------------:|:---------:|:--------:|:--------:|
|        Training        |    544     |     66      |  13,524  |  492,057  |
|        Validation        |    137     |     66      |  3,488  |  121,750  |
|        Test        |    96     |     66      |  2,453  |  78,995  |


## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## Citation🚩

```
@inproceedings{KIRE,
  author    = {Xinyi Wang and
               Zitao Wang and
  	       Weijian Sun and
               Wei Hu},
  title     = {Enhancing Document-level Relation Extraction by Entity Knowledge Injection},
  booktitle = {ISWC},
  year      = {2022}
}
```
