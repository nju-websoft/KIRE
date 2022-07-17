# SSAN-KIRE

This is the KIRE model based on the [SSAN](https://github.com/BenfengXu/SSAN) .

## Requirements

* Python (tested on 3.6.2)
* CUDA (tested on 11.3)
* [PyTorch](http://pytorch.org/) (tested on 1.10.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 2.8.0)
* numpy
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

Train the SSAN and KIRE + SSAN model on DocRED with the following command:

```bash
>> sh train_base_docred.sh  # for SSAN_BERT_base model
>> sh train_kire_docred.sh  # for SSAN_BERT_base + KIRE model 
```

The program will generate a test file `result.json` in the official evaluation format in `./checkpoints/bert_base`. You can compress and submit it to Colab for the official test score.

### DWIE

Train the SSAN and KIRE + SSAN model on DWIE with the following command:

```bash
>> sh train_base_dwie.sh  # for SSAN_BERT_base model
>> sh train_kire_dwie.sh  # for SSAN_BERT_base + KIRE model 
```

## Saving and Evaluating Models

You can save the model by setting the `--output_dir` and `--checkpoint_dir` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--do_predict` argument and don't set the `--do_train` argument, then the code will skip training and evaluate the saved model on benchmarks. 

