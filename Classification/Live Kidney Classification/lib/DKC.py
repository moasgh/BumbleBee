import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from time import time
import os
import seaborn as sns
import lib.transformers as trf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mysql.connector
import nltk
from nltk.util import ngrams
from collections import Counter
import base64

from torch.utils.tensorboard import SummaryWriter

class DataHandler():
    def __init__(self):
        self.user = None
        self.password = None
        self.host = None
        self.database = None

    def get_data_from_mysql(self,mysql_info,  sql, columns):
        
        self.user = mysql_info['user']
        self.password = mysql_info['password']
        self.host = mysql_info['host']
        self.database = mysql_info['database']
        cnx = mysql.connector.MySQLConnection(user= self.user , password = self.password, host = self.host, database = self.database)
        cursor = cnx.cursor()
        cursor.execute(sql)
        data = cursor.fetchall()
        df =  pd.DataFrame(data = data, columns = columns)
        return df
    
    def __extract_ngrams(self, data, num):
        n_grams = ngrams(nltk.word_tokenize(data), num)
        return [' '.join(grams) for grams in n_grams]

    def get_data_from_my_sql_prepare(self , sql , mysql_info, columns):
        data =  self.get_data_from_mysql(mysql_info , sql , columns)
        data =  trf.cleaner(data)
        data.clean()
        data.tokenize()
        data.stats()
        data.documents['2grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),2))     
        data.documents['3grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),3))     
        data.documents['4grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),4)) 
        return data

    def prepare_data(self, data):
        data = trf.cleaner(data)
        data.clean()
        data.tokenize()
        data.stats()
        data.documents['2grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),2))     
        data.documents['3grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),3))     
        data.documents['4grams'] = data.documents['stem_words'].apply(lambda x: self.__extract_ngrams(' '.join(x),4)) 
        return data
    
    def b64tostring(s):
        s = s.encode('utf8')
        b_base64 = base64.b64decode(s)
        return b_base64.decode('utf8')

    def load_data(self,document):
        data = trf.cleaner()
        data.set_documents(document)
        return data

    def balance(self, X , Y):
        documents_length = trf.cleaner.doc_length_distibution(X , 'stem_words')
        Target_idx = {}
        for idx,y in zip(X.index,Y):
            if y in Target_idx:
                Target_idx[y].append(idx)
            else:
                Target_idx[y] = [idx]
        idx_TargetLength = {}
        for idx,y,l in zip(X.index,Y, documents_length):
            idx_TargetLength[idx] = [y,l]
        length_TargetIdxs = {}
        for idx,(y,l) in idx_TargetLength.items():
            if l in length_TargetIdxs:
                length_TargetIdxs[l].append((idx,y))
            else:
                length_TargetIdxs[l] = [(idx,y)]
        # balance 
        Length_BalanceTargetIDX_Trains = {}
        Length_BalanceTargetIDX_remain = {}
        for l, list_items in length_TargetIdxs.items():
            Length_BalanceTargetIDX_Trains[l] = []
            Length_BalanceTargetIDX_remain[l] = []
            list_items = sorted(list_items,key=lambda x: x[1])
            labels_count = Counter([ label for _,label in list_items ])
            balance_min_length = min(labels_count.values())
            offset = 0
            if len(labels_count)>1:
                for label in labels_count.keys():
                    Length_BalanceTargetIDX_Trains[l].append(list_items[offset:offset+balance_min_length])
                    if(balance_min_length < labels_count[label]):
                        Length_BalanceTargetIDX_remain[l].append(list_items[offset+balance_min_length:])
                    offset += labels_count[label]
        X_train_idx_y = []
        for _,records in Length_BalanceTargetIDX_Trains.items():
            for rs in records:
                X_train_idx_y.extend(rs)
        
        return X_train_idx_y

class DKC(nn.Module):
    
    def __init__(self, params, vocabs):
        super(DKC,self).__init__()
        
        self.vocabs = vocabs
        self.embedding_type = params['embbeding_type']
        self.__params = params
        self.__params['vocab_size'] = len(self.vocabs)
        
        self.embeding = nn.Embedding(self.__params['vocab_size'], self.__params['embedding_dim'])
        self.rnn = nn.LSTM(self.__params['embedding_dim'],self.__params['hidden_dim'])
        self.hidden_2class = nn.Linear(self.__params['hidden_dim'],self.__params['output_number'])
        #self.init_weights()
        self = self.cuda(self.__params['cuda']['device']) if self.__params['cuda']['isactive'] else self 
    
    def init_weights(self):            
        for m in self.rnn.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                print(m)
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.embeding.weight)
        torch.nn.init.xavier_uniform_(self.hidden_2class.weight)

    def forward(self, comment):
        
        embed = self.embeding(comment)
        output, _ = self.rnn(embed)
        output = self.hidden_2class(output)
        output = torch.sum(output,1)
        
        return output
    
    def loads(self,state_dict):
        
        self.load_state_dict(state_dict)
    
    def save(self, location , extra_params = None):
        """
        location must contain the name of the file
        extra_params can be any parameters such as: 
            epoch
            optimizer_state
            prediction
            f1_score
        """
        
        current_model_state = {
                    'model_state_dict' : self.state_dict(),
                    'params': {
                        'embbeding_type': self.__params['embbeding_type'],
                        'vocab_size': self.__params['vocab_size'],
                        'embedding_dim': self.__params['embedding_dim'],
                        'output_number': self.__params['output_number'],
                        'hidden_dim': self.__params['hidden_dim'],
                        'hop': self.__params['hop'],
                        'lr': self.__params['lr'],
                        'cuda': self.__params['cuda'],
                        'batch_size': self.__params['batch_size']} }
        if extra_params is not None:
            for key,val in extra_params.items():
                current_model_state[key] = val
        
        
        torch.save(current_model_state, location)
        
    def __get_vector(self,row):
        
        vec = []
        
        for w in row:
            if w in self.vocabs:
                vec.append(self.vocabs[w])
            else:
                vec.append(self.vocabs['UNK'])
        return vec
            
    def __padding_data(self,data,target, column):
        
        data_tensor = []
        max_length = 0
        
        for i,row in data.iterrows():
            vector = self.__get_vector(row[column])
            length = len(vector)
            max_length = length if length > max_length else max_length
            data_tensor.append(vector)
        
        data_tensor_pad = np.zeros((data.shape[0] , max_length))
        for i,s in enumerate(data_tensor):
            data_tensor_pad[i] = s + [self.vocabs['PAD']]*(max_length - len(s))
            
        if target is not None:
            target = torch.LongTensor(target)
            
        data = torch.LongTensor(data_tensor_pad)
        return data, target
    
    def __batch_preperation(self,data,target):
        
        vec_data , target_data = None, None
        
        if(self.embedding_type.startswith('char_gram') ):
            vec_data , target_data = self.__padding_data(data, target, 'stem_words')
        else:
            vec_data , target_data = self.__padding_data(data, target, self.embedding_type)
            
        batch_size = self.__params['batch_size']
        hop = self.__params['hop']
        
        batch_data = []
        if batch_size > 0:
            for i in range(0, len(vec_data) , hop):
                if target_data is not None:
                    batch_data.append([vec_data[i:i+batch_size], target_data[i:i+batch_size]])
                else:
                    batch_data.append([vec_data[i:i+batch_size], None])
        else:
            batch_data.append((vec_data, target_data))
            
        return batch_data
        
    def trains(self, data, target, train_params = {}):
        
        if 'optimizer' in train_params and 'criterion' in train_params:
            result = []
            __optimizer = train_params['optimizer']
            __criterion = train_params['criterion']
            best_model = None
            Batch_Data =  self.__batch_preperation(data, target)
            
            cuda = self.__params['cuda']
            
            for e in range(1,self.__params['epoch']+1):
                train_acc_list = []
                train_loss_list = []
                train_acc , train_loss = 0, 0

                for b_data,b_target in Batch_Data:
                    b_data = b_data.to(device='cuda:' + str(cuda['device'])) if cuda['isactive'] else b_data
                    b_target = b_target.to(device='cuda:' + str(cuda['device'])) if cuda['isactive'] else b_target
                    output = self.forward(b_data)
                    loss = __criterion(output,b_target)
                    train_loss_list.append(loss.item())
                    loss.backward()
                    __optimizer.step()
                    train_acc_list.append((output.argmax(1) == b_target).sum().item() / output.shape[0])
                train_acc = np.mean(train_acc_list)
                train_loss = np.mean(train_loss_list)
                
                if 'eval' in train_params:

                    model_location = self.__params['save_model_location_dir']

                    with torch.no_grad():

                        eval_data = train_params['eval']['data']
                        eval_target = train_params['eval']['target']

                        predicted, true_value = self.evals(eval_data,eval_target)

                        macro_f1_score = f1_score(true_value, predicted , average='macro')
                        micro_f1_score = f1_score(true_value, predicted , average='micro')
                        test_acc = (np.array(predicted) == np.array(true_value)).sum().item() / len(true_value)

                        #test_loss = np.mean(test_loss_list)
                        print('train acc : {:.3f}  test macro-f1: {:.3f}  test micro-f1: {:.3f}'
                              .format(train_acc, macro_f1_score, micro_f1_score)  , end = "\r") 
                        result.append([e, train_acc, train_loss, micro_f1_score, macro_f1_score, test_acc, 0])

                        if best_model is None or macro_f1_score > best_model['f1_score']:

                            best_model = {'epoch': e ,
                                        'optimizer_state_dict' : __optimizer.state_dict() , 
                                        'f1_score': macro_f1_score, 
                                        'predict': [true_value, predicted] }
                            print()
                            print('best model f1-score:',best_model['f1_score'])
                            best_model['vocabs'] = self.vocabs
                            print('saving ...')
                            if not os.path.exists(model_location):
                                os.mkdir(model_location)
                                self.save(model_location+'/best.pth', extra_params=best_model)
                            else:
                                #index_model = len(os.listdir(model_location))+1
                                os.remove(model_location+'/best.pth')
                                self.save(model_location+'/best.pth', extra_params=best_model)
            return result
                    
    def evals(self,data, target):
        
        cuda = self.__params['cuda']
        
        # prepare the batch based on batch size
        Batch_Data_eval = self.__batch_preperation(data, target)
        
        predicted = []
        target_true = []
        for b_data,b_target in Batch_Data_eval:
            b_data = b_data.to(device='cuda:' + str(cuda['device'])) if cuda['isactive'] else b_data
            b_target = b_target.to(device='cuda:' + str(cuda['device'])) if cuda['isactive'] else b_target
            output_test = self.forward(b_data)

            if cuda['isactive']:
                predicted.extend(output_test.argmax(1).cpu().numpy())
                target_true.extend(b_target.cpu().numpy())
            else:
                predicted.extend(output_test.argmax(1))
                target_true.extend(b_target)
                
        return predicted,target_true
    
    def predicts(self, data):
        
        cuda = self.__params['cuda']
        
         # prepare the batch based on batch size
        batch_data = self.__batch_preperation(data, None)

        predicted = []
        for b_data,_ in batch_data:
            b_data = b_data.to(device='cuda:' + str(cuda['device'])) if cuda['isactive'] else b_data
            
            output_test = self.forward(b_data)

            if cuda['isactive']:
                predicted.extend(output_test.argmax(1).cpu().numpy())
            else:
                predicted.extend(output_test.argmax(1))

        return predicted

class DeployModel():
    def __init__(self):
        self.data = None

    def load_model(self,models_location, data_location, mode = 'eval'):
        data = pd.read_pickle(data_location)
        self.data = data
        target = []
        if 'class' in data.columns:
            target = data['class'].tolist()
        if mode=='eval' and len(target) == 0:
            print('Eval model need target')
            return
        model_name = ""
        for d in os.listdir(models_location):
            if 'pth' in d:
                model_name = d

        model_location = models_location+'/'+model_name
        # to run the model on GPU just change the cpu to cuda:0 
        model_meta_data = torch.load(model_location, map_location='cpu')
        print(model_meta_data['params'])
        model_load_test = DKC(model_meta_data['params'], model_meta_data['vocabs'])
        model_load_test.loads(model_meta_data['model_state_dict'])
        if mode == 'eval':
            eval = model_load_test.predicts(data)
            self.data['eval'] = eval
            return model_load_test.evals(data, target)
        elif mode == 'pred':
            pred = model_load_test.predicts(data)
            self.data['pred'] = pred 
            return pred, target

    def get_model(self,models_location, map_location = 'cuda:0'):
        model_name = ""
        for d in os.listdir(models_location):
            if 'pth' in d:
                model_name = d
        model_location = models_location+'/'+model_name
        model_meta_data = torch.load(model_location, map_location)
        print(model_meta_data['params'])
        model = DKC(model_meta_data['params'], model_meta_data['vocabs'])
        return model

    def draw_performance(self,models_location, title=""):
        #model_meta_data = torch.load(model_location, map_location='cuda:0')

        # performance_data = pd.DataFrame(data = models_location+'results.csv' , 
        #                         columns = ['epoch' , 'train_acc', 'train_loss', 
        #                                     'test_micro_f1', 'test_macro_f1', 'test_acc', 'test_loss'])
        performance_data = pd.read_csv(models_location+'/result.csv')
        sns.lineplot(performance_data['epoch'],performance_data['train_acc'],label='acc train')
        sns.lineplot(performance_data['epoch'],performance_data['test_acc'],label='acc test')
        sns.lineplot(performance_data['epoch'],performance_data['test_macro_f1'],label='test f1')
        plt.title(title, fontsize=20)
        plt.xlabel("Epoch" , fontsize=15)
        plt.ylabel("F1-Score / Accuracy", fontsize=15)
        
        plt.hlines(performance_data['test_macro_f1'].max() , 0 , performance_data['epoch'].max() , linestyles='--' , color = 'red')
        plt.vlines(np.argmax(performance_data['test_macro_f1']) , 0 , 1, linestyles='--', color = 'red')
        print('best f1 score:', performance_data['test_macro_f1'].max())
        print('best epoch: ' , np.argmax(performance_data['test_macro_f1']))


            




