# Netizens' emotion recognition during the epidemic

Used PaddleHub and ERNIE to realize emotion recognition of texts during the epidemic

## Project Background

The outbreak of 2019-nCoV-infected pneumonia has had an important impact on all aspects of people's life and production, and has aroused widespread attention of domestic public opinion. Many netizens have participated in discussions on topics related to the epidemic. In order to help the government grasp the real social public opinion situation, scientifically and efficiently do a good job of prevention and control propaganda and public opinion guidance, I carried out the task of identifying netizens' emotions on topics related to the epidemic.

## Project Description

The data set is based on 230 subject keywords related to "new crown pneumonia" for data collection. A total of 1 million Weibo data were captured from January 1, 2020 to February 20, 2020, and 100,000 of them were collected. The data is manually labeled, and the labels are divided into three categories: 1 (positive), 0 (neutral) and -1 (negative). The content of Weibo includes text, pictures, videos, etc. This program selects text content for emotion recognition of Weibo content, and uses Macro-F1 value for scoring.

## Data Analysis

Let's unzip the data set first.

```shell
cd data/data22724 && unzip test_dataset.zip
```

Since the data adopts GB2312 encoding, the data is read out first, converted to UTF-8 encoding and then rewritten to facilitate subsequent use and processing of the Pandas library.

```python
# Transcoding
def re_encode(path):
    with open(path, 'r', encoding='GB2312', errors='ignore') as file:
        lines = file.readlines()
    with open(path, 'w', encoding='utf-8') as file:
        file.write(''.join(lines))
        
re_encode('data/data22724/nCov_10k_test.csv')
re_encode('data/data22724/nCoV_100k_train.labled.csv')
```

### Data preview

Read the data, check the data size, column name, you can see:

The training set contains 100,000 pieces of data, and the test set contains 10,000 pieces of data

The data includes Weibo id, Weibo release time, publisher account number, Weibo Chinese content, Weibo pictures, Weibo videos, and emotional orientation. The Chinese content of Weibo is the training and test data we will use, and the emotional tendency is the problem label.

```python
# Read data
import pandas as pd
train_labled = pd.read_csv('data/data22724/nCoV_100k_train.labled.csv', engine ='python')
test = pd.read_csv('data/data22724/nCov_10k_test.csv', engine ='python')
```

```python
print(train_labled.shape)
print(test.shape)
print(train_labled.columns)
```

### Label Distribution

Question labels are divided into three categories, namely: 1 (positive), 0 (neutral) and -1 (negative), and their distribution is as follows.

It can be seen that the number of neutral data accounts for more than half, followed by positive data and the least negative data.

In addition, the data also contains a small number of unknown labels, which we regard as abnormal data to eliminate.

```python
# Label Distribution
%matplotlib inline
train_labled['情感倾向'].value_counts(normalize=True).plot(kind='bar');
```

Clear abnormal label data

```python
train_labled = train_labled[train_labled['情感倾向'].isin(['-1','0','1'])]
```

### Text length

The maximum text length of the training set is 241, and the average is 87.

```python
train_labled['微博中文内容'].str.len().describe()
```

## Program Introduction

​	If we only consider Weibo text data, the problem we need to solve is actually the text classification problem.

​	With the release of models such as ELMo and BERT in 2018, the NLP field has entered an era of "making miracles vigorously". Using a deep model for unsupervised pre-training on a large-scale corpus and fine-tuning it on the downstream task data can achieve good results. The tasks that used to require repeated parameter adjustment and elaborate structure design can now be solved simply by using larger pre-training data and deeper models.

​	I use the PaddleHub pre-training model fine-tuning tool produced by Baidu to quickly build the solution. For the model, the ERNIE model is chosen.

​	ERNIE 1.0 learns real-world semantic knowledge by modeling words, entities and entity relationships in massive data. Compared with BERT learning the original language signal, ERNIE directly models the prior semantic knowledge unit, which enhances the semantic representation ability of the model.

​	The pre-training model management and migration learning tools produced by PaddlePaddle can easily obtain pre-training models under the PaddlePaddle ecosystem, and complete model management and one-click prediction. With the use of Fine-tune API, migration learning can be quickly completed based on large-scale pre-training models, so that pre-training models can better serve user-specific applications.

The use process is as follows:

1) Organize the data into a specific format

2) Define the Dataset data class

3) Load the model

4) Build the reader data reading interface

5) Determine the finetune training strategy

6) Configure finetune parameters

7) Determine the task and start finetune (training)

8) Forecast

## Data Collation

### Update paddlehub

```shell
pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```

We divide the data into training set and validation set with a ratio of 8:2, and then save it as a text file. The two columns need to be separated by a tab separator.

```python
# Divide the verification set and save the format
from sklearn.model_selection import train_test_split

train_labled = train_labled[['微博中文内容', '情感倾向']]
train, valid = train_test_split(train_labled, test_size=0.2, random_state=2020)
train.to_csv('/home/aistudio/data/data22724/train.txt', index=False, header=False, sep='\t')
valid.to_csv('/home/aistudio/data/data22724/valid.txt', index=False, header=False, sep='\t')
```

### Custom data loading

To load a text-based custom data set, the user only needs to inherit the base class BaseNLPDatast and modify the storage address and category of the data set.

Here we do not have a labeled test set, so test_file directly replaces "valid.txt" with a validation set. Custom data loading

```python
# Custom data set
import os
import codecs
import csv

from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

class MyDataset(BaseNLPDataset):
    """DemoDataset"""
    def __init__(self):
        # Data set storage location
        self.dataset_dir = "/home/aistudio/data/data22724"
        super(MyDataset, self).__init__(
            base_path=self.dataset_dir,
            train_file="train.txt",
            dev_file="valid.txt",
            test_file="valid.txt",
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            # Data set category collection
            label_list=["-1", "0", "1"])

dataset = MyDataset()
for e in dataset.get_train_examples()[:3]:
    print("{}\t{}\t{}".format(e.guid, e.text_a, e.label))
```

### Load the model

Here we choose the Chinese pre-training model of ERNIE 1.0.

```python
# Load the model
import paddlehub as hub
module = hub.Module(name="ernie")
```

### Build Reader

Construct a reader for text classification. The reader is responsible for preprocessing the data of the dataset. First, the text is segmented, and then organized in a specific format and input to the model for training.

The maximum sequence length can be modified by max_seq_len. If the sequence length is insufficient, max_seq_len will be filled in by padding. If the sequence length is greater than this value, the sequence length will be truncated to max_seq_len, here we set it to 128.

```python
# Build Reader
reader = hub.reader.ClassifyReader(
    dataset=dataset,
    vocab_path=module.get_vocab_path(),
    sp_model_path=module.get_spm_path(),
    word_dict_path=module.get_word_dict_path(),
    max_seq_len=128)
```

### Finetune Strategy

Choose a migration optimization strategy

Here we set the maximum learning rate learning_rate=5e-5.

The weight decay is set to weight_decay=0.01, which avoids model overfitting.

The training preheating ratio is set to warmup_proportion=0.1, so that the learning rate in the first 10% of the training steps will gradually increase to learning_rate

```python
# Finetune Strategy
strategy = hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    warmup_proportion=0.1,
    learning_rate=5e-5)
```

### Run configuration

Set epoch, batch_size, model storage path and other parameters during training.

Here we set the number of training rounds num_epoch = 1, the model save path checkpoint_dir="model", every 100 rounds (eval_interval), the validation set is verified and scored, and the optimal model is saved.

```python
# Run configuration
config = hub.RunConfig(
    use_cuda=True,
    num_epoch=1,
    checkpoint_dir="model",
    batch_size=32,
    eval_interval=100,
    strategy=strategy)
```

### Build Finetune Task

For text classification tasks, we need to obtain the output of the pooling layer of the model, followed by a fully connected layer to achieve classification.

Therefore, we first obtain the context of the module, including input and output variables, and obtain the output of the pooling layer as a text feature. Then connect a fully connected layer to generate Task.

The evaluation index is F1, so set metrics_choices=["f1"]

```
# Finetune Task
inputs, outputs, program = module.context(
    trainable=True, max_seq_len=128)

# Use "pooled_output" for classification tasks on an entire sentence.
pooled_output = outputs["pooled_output"]

feed_list = [
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name,
]

cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=dataset.num_labels,
        config=config,
        metrics_choices=["f1"])
```

### Start finetune

We can start model training by using the finetune_and_eval interface. During the finetune process, the model effect will be periodically evaluated.

```python
# finetune
run_states = cls_task.finetune_and_eval()
```

### Prediction

After fine tune is completed, call the predict interface to complete the prediction
The forecast data format is a two-dimensional list:
[['The first text'], ['The second text'], [...], ...]

```python
# Prediction
import numpy as np
    
inv_label_map = {val: key for key, val in reader.label_map.items()}

# Data to be prdicted
data = test[['微博中文内容']].fillna(' ').values.tolist()

run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
```

### Generate results

```python
# Generate prediction results
proba = np.vstack([r[0] for r in results])
prediction = list(np.argmax(proba, axis=1))
prediction = [inv_label_map[p] for p in prediction]
        
submission = pd.DataFrame()
submission['id'] = test['微博id'].values
submission['id'] = submission['id'].astype(str) + ' '
submission['y'] = prediction
np.save('proba.npy', proba)
submission.to_csv('result.csv', index=False)
submission.head()
```

## Summary

Thank you TV

Thank you all the TV