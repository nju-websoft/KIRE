# KIRE
Enhancing Document-level Relation Extraction by Entity Knowledge Injection, ISWC 2022


## Models
Source code for KIRE models based on four [baselines](https://github.com/thunlp/DocRED), [GLRE](https://github.com/nju-websoft/GLRE),  [SSAN](https://github.com/BenfengXu/SSAN) and [ATLOP](https://github.com/wzhouad/ATLOP)  are saved in "./B4+KIRE",  "./GLRE+KIRE", "./SSAN+KIRE"  and "./ATLOP+KIRE" respectively.


### B4+KIRE

In "./B4+KIRE", we provide the following directories:

* configs: the code for running the model;
* checkpoints: store the model after train;
* prepro_data:  the preprocess data;
* model: the models include CNN, LSTM, BiLSTM, Context-aware;
* knowledge_injection_layer: the knowledge injection module;
* scripts: different code files corresponding to the sh files in the home directory;


### GLRE+KIRE

In "./GLRE+KIRE", we provide the following directories:

* configs: different configs used for experiments;
* data: datasets and corresponding data loading code;
* data_processing: the preprocess code for datasets;
* knowledge_injection_layer: the knowledge injection module;
* scripts: different code files corresponding to the sh files in the home directory;
* other directories including "models", "nnet", "utils" contain source code of GLRE model.


### SSAN+KIRE

In "./SSAN+KIRE",we provide the following directories:

* data: datasets and corresponding generate code;
* knowledge_injection_layer: the knowledge injection module;
* checkpoints: store the model after train
* pretrained_lm: store the pretrained model as 
* other directories or files including "model", "run_docred.py", "dataset.py" contain source code of SSAN model.

  

### ALTOP+KIRE

In "./ALTOP+KIRE", we provide the following directories:

* data: datasets and corresponding generate code;
* knowledge_injection_layer: the knowledge injection module;
* scripts: different sh files used for experiments under different settings;





