# KIRE
This is the KIRE model based on the [ATLOP](https://github.com/wzhouad/ATLOP).

## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 2.11.0)
* numpy
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm
* yamlordereddictloader (tested on 0.4.0)
* scipy (tested on 1.5.2)

## Data

Download processed data from [figshare](https://figshare.com/articles/dataset/Processed_data/14602203) and put it into the `./data` directory.

## Pretrained models
Download pretrained autoencoder model from [figshare](https://figshare.com/articles/dataset/re_model/14602185) and put it into the `./knowledge_injection_layer/ae_results` directory.
Download pretrained language model from [huggingface](https://huggingface.co/bert-base-uncased) and put it into the `./bert_base` directory. 

## Run
### DocRED
Train the ATLOP and KIRE + ATLOP model on DocRED with the following command:

```bash
>> sh scripts/run_docred_bert.sh  # for ATLOP_BERT_base model
>> sh scripts/run_docred_bert_kire.sh  # for ATLOP_BERT_base + KIRE model 
```

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.

### DWIE

Train the ATLOP and KIRE + ATLOP model on DWIE with the following command:

```bash
>> sh scripts/run_dwie_bert.sh  # for ATLOP_BERT_base model
>> sh scripts/run_dwie_bert_kire.sh  # for ATLOP_BERT_base + KIRE model 
```

## Saving and Evaluating Models
You can save the model by setting the `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks. 
