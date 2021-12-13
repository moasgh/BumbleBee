# BumbleBee
<img src="https://user-images.githubusercontent.com/25641555/76114333-d7a63480-5fb3-11ea-96e1-8d2ff27c4a7f.png" width="128" height="200" />


# Deployment
```python
import handlers as hd
import os
```

```
Set the status of Model Handler to 'load'
```

```python
modelhandler = hd.ModelHandler('load',None)
```

```
modelhandler.load_model_predict(
model_name,
model_path,
data_path,
output_path_filename=None,
output_type='flat',
sentence_seperator='\n',
map_location=0,
verbos=False,
)
---------------------------------------------------------------------------------------
model_name = it needs to be from this list 
'rnn_two_crf_par' , 
'rnn_two_crf', 
'rnn_two_crf_seq' , 
'rnn_two_crf_seq2', 
'rnn_single_crf' also the check point should be compatible with the implemented model otherwise will raise an error
---------------------------------------------------------------------------------------
model_path = location of model exp: ../models/model.ckp
No restiction on extention of the file (ckp = check point)
---------------------------------------------------------------------------------------
data_path = location of test data
---------------------------------------------------------------------------------------
result_location = is the directory that we want to save the result of the rest this directory if is not exist will be generated
---------------------------------------------------------------------------------------
map_location = is an integer that let to assign the model to any gpu when you have multiple gpu (default is 0)
---------------------------------------------------------------------------------------
```

```python
output_type = ['flat' , 'group']
modelhandler.load_model_predict('rnn_two_crf_seq',
                            'models/Disease/model',
                            'datasets/test.txt',
                            'datasets/result_test',
                            'flat',
                            '\n', 0, True)
```
```
CUDA is avaiable : True
Unavailable CUDA might cause an error.
Model is mapped to : CUDA 0 ACTIVE-DEVICE to host the calculations : CUDA  0
The difference between these two might cause an error. Please change the ACTIVE_DEVICE in util.py if it is necessary.
loading models/Disease/model
saved model: epoch = 500, loss = 0.005174
{'UNIT': 'word', 'TASK': None, 'RNN_TYPE': 'LSTM', 'NUM_DIRS': 2, 'NUM_LAYERS': 10, 'BATCH_SIZE': 32, 'EMBED': {'char-rnn': 100}, 'HIDDEN_SIZE': 300, 'DROPOUT': 0.3, 'LEARNING_RATE': 0.0002, 'EVAL_EVERY': 10, 'SAVE_EVERY': 10, 'EPOCH': 500, 'model_name': 'rnn_two_crf_seq2', 'output_path': 'rnn_two_crf_seq2_char-rnn_NCBI-disease-IOB', 'EMBED_SIZE': 100, 'HRE': False}
rnn_two_crf_seq2(
  (rnn_ner): rnn(
    (embed): embed(
      (char_embed): rnn(
        (embed): Embedding(62, 100, padding_idx=0)
        (rnn): GRU(100, 50, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
      )
    )
    (rnn): LSTM(100, 150, num_layers=10, batch_first=True, dropout=0.3, bidirectional=True)
    (out): Linear(in_features=300, out_features=5, bias=True)
  )
  (crfner): crf()
  (rnn_iob): LSTM(5, 150, num_layers=10, batch_first=True, dropout=0.3, bidirectional=True)
  (out): Linear(in_features=300, out_features=6, bias=True)
  (crfiob): crf()
)
```

Natural Language Processing , LSTM , CNN, NER

# References

Xuezhe Ma, Eduard Hovy. 2016. <a href = "https://arxiv.org/abs/1603.01354"> End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. </a> arXiv:1603.01354.
Code : https://github.com/threelittlemonkeys/lstm-crf-pytorch

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, Jaewoo Kang 2019. <a href=""> BioBERT: a pre-trained biomedical language representation model for biomedical text mining. </a> arXiv:1901.08746v4.

Code : https://github.com/dmis-lab/biobert

