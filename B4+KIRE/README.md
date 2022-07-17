# B4-KIRE

This is the KIRE model based on four baseline.

## Requirements

* Python (tested on 3.6.2)
* CUDA (tested on 11.3)
* [PyTorch](http://pytorch.org/) (tested on 1.10.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 2.8.0)
* numpy
* ujson
* scipy (tested on 1.5.2)

## Data

Download processed data from [figshare](https://figshare.com/articles/dataset/Processed_data/14602203) and put it into the `./prepro_data` directory.

## Pretrained models

Download pretrained autoencoder model from [figshare](https://figshare.com/articles/dataset/re_model/14602185) and put it into the `./knowledge_injection_layer/ae_results` directory.

## Run

### DocRED

Train the baseline  model and baseline model + KIRE model on DocRED with the following command:

For example, we train and test the CNN and CNN+KIRE.

```bash
>> cd ./script/base_docred
>> sh train_CNN_base_docred.sh  # for CNN model 
>> cd ../kire_docred
>> sh train_CNN_kire_docred.sh # for CNN+Kire model
>> sh test_CNN_kire_docred.sh # test CNN+Kire model
```

The program will generate a test file `result.json` in the official evaluation format in `./checkpoints`. You can compress and submit it to Colab for the official test score.

### DWIE

Train the baseline  model and baseline model + KIRE model on DWIE with the following command:

For example, we train and test the CNN and CNN+KIRE.

```bash
>> cd ./script/base_dwie
>> sh train_CNN_base_dwie.sh  # for CNN model 
>> cd ../kire_dwie
>> sh train_CNN_kire_dwie.sh # for CNN+Kire model
>> sh test_CNN_kire_dwie.sh # test CNN+Kire model
```

