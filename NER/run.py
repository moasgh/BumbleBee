from handlers import *
from utils import *
import re
import os
embeds = [ {"char-rnn": 100} , {"char-cnn": 100} , {"lookup": 100} , {"sae": 100}]
models = ['rnn_two_crf_par' , 'rnn_two_crf', 'rnn_two_crf_seq' , 'rnn_two_crf_seq2', 'rnn_single_crf']
for dbname in os.listdir('datasets/'):
    db_name = dbname
    for embed in embeds:
        for model in models:
            output_path = model + '_' + list(embed.keys())[0] + '_' + db_name
            params = {'UNIT' : "word" # unit of tokenization (char, word, sent)
                ,'TASK' : None # task (None, word-segmentation, sentence-segmentation)
                ,'RNN_TYPE' : "LSTM" # LSTM or GRU
                ,'NUM_DIRS' : 2 # unidirectional: 1, bidirectional: 2
                ,'NUM_LAYERS' : 10
                ,'BATCH_SIZE' : 32
                ,'EMBED' : embed # embeddings (char-cnn, char-rnn, lookup, sae)
                ,'HIDDEN_SIZE' : 300
                ,'DROPOUT' : 0.3
                ,'LEARNING_RATE' : 2e-4
                ,'EVAL_EVERY' : 10
                ,'SAVE_EVERY' : 10
                ,'EPOCH' : 500
                , 'model_name' : model
                , 'output_path' : output_path}
            if output_path not in os.listdir():
                BC2GM_IOB = ModelHandler('train' , params , db_location= 'datasets/' + db_name + "/", load_percentage=1)
                print(params)
                print('Model is Runnign on GPU number =' + str(ACTIVE_DEVICE))
                print("output path is => ", output_path)
                BC2GM_IOB.train(output_path=output_path)
            else:
                train_checkpoints = os.listdir(output_path)
                train_checkpoints = sorted(train_checkpoints , key = lambda x : int(re.findall('epoch\d+' , x)[0].replace('epoch','') if re.findall('epoch\d+',x) else 0))
                last_traind_epoch = train_checkpoints[-1]
                last_epoch = int(re.findall('epoch\d+' , last_traind_epoch.lower())[0].replace('epoch','')) 
                if params["EPOCH"] == last_epoch:
                    print('Training of ' + output_path + 'is Done.')
                else:
                    BC2GM_IOB = ModelHandler('train' , params , db_location= 'datasets/' + db_name + "/", load_percentage=1)
                    print(params)
                    print('Model is Runnign on GPU number =' + str(ACTIVE_DEVICE))
                    print("output path is => ", output_path)
                    BC2GM_IOB.train(output_path=output_path , retrain=True , model_path=output_path + '/' +  last_traind_epoch)


