import sys
import re
from time import time
import os
from glob import glob
from os.path import isfile
#from parameters import *
from collections import defaultdict , Counter
from sklearn.metrics import roc_auc_score , roc_curve , auc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token
O = 'o'
I = 'i'
B = 'b'


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
O_IDX = 3
B_IDX = 4
I_IDX = 5


CUDA = torch.cuda.is_available()
torch.manual_seed(0) # for reproducibility
ACTIVE_DEVICE = 0

Tensor = lambda *x: torch.FloatTensor(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.FloatTensor
LongTensor = lambda *x: torch.LongTensor(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.LongTensor
randn = lambda *x: torch.randn(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.randn
zeros = lambda *x: torch.zeros(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.zeros

KEEP_IDX = False # use the existing indices when adding more training data
NUM_DIGITS = 4 # number of digits to print


def normalize(x):
    # x = re.sub("[\uAC00-\uD7A3]+", "\uAC00", x) £ convert Hangeul to 가
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def transform(text):
    def normalizespaces(x):
        return  re.sub('\s+' , '' , x.group(1))
    def normalizeparenthesis(x):
        return x.group(1)+re.sub('([^\s\w])' , r' \1 ' , x.group(2))+x.group(3)
    def noncharacterspcae(x):
        return ' ' +x.group(1).strip()+ ' '
    def commanormalization(x):
        return re.sub(r'(\,)' , r' \1 ' , x.group(1))
    def especialwordnormalization(x):
        return re.sub(r'(\.)' , r' \1 ' , x.group(1))
    transforms = [
        # this will find all the nonchracters and remove all the spaces to normalize
        ('sub',r'((\s\W\s?)|(\s?\W\s))' , normalizespaces, False) ,
        # create space in pranthesis to avoid them in order to split based on dot for each sentence
        ('sub',r'([\(\[])(.+?)([\)\]])' , normalizeparenthesis, False),
        # create space in everything except dot
        ('sub' , r'([^\w\d\.\s\,])' , noncharacterspcae, False),
        # Comma normalization becuase some of the numbers are seperated by comma  
        ('sub' , r'([^0-9]\,([^0-9])?|[^0-9]?\,([^0-9]))' , commanormalization, False),
        # this is specific for those words where they are name of something seperated by .
        # such as E.bola or U.S.A
        # True it means need to be recursive becuase the action might create new sample
        # such as U.S.A => U . S.A => U . S . A
        ('sub',r'(\s[a-zA-z]{1}\.(\s+)?)' , especialwordnormalization, True),
        # split based on dot to create sentences
        ('split' , r'[\.\!\?](?!\d)(?!\s)'),
        # normalize created sentences
        ('sub',r'((\s\W\s?)|(\s?\W\s))' , normalizespaces, False),
        # create space for nonchartacters for future tokenization
        # this will undrestand the unicode also
        ('sub' , r'([^\w\d\.\s\,])' , noncharacterspcae, False),
        # Comma normalization becuase some of the numbers are seperated by comma  
        ('sub' , r'([^0-9]\,([^0-9])?|[^0-9]?\,([^0-9]))' , commanormalization, False),
        ('sub' , r'(\w\.(?!\d+))' , especialwordnormalization , False)
    ]
    sentences = []
    for tran in transforms:
        if tran[0] == 'sub':
            if sentences:
                sentences = [re.sub(tran[1] , tran[2] , s) for s in sentences]
            else:
                text = re.sub(tran[1] , tran[2] , text)
                #print(tran[0])
                #print(text)
                # this is where we mentioned if there need to be iterate
                if tran[3]:
                    while re.findall(tran[1] , text):
                        text = re.sub(tran[1] , tran[2] , text)
        elif tran[0] == 'split':
            # we add dot to the end of each sentence
            # we lower each sentence
            sentences = [ s.lower() + ('' if s.strip().endswith('.') else ' .') for s in re.split(tran[1] , text) if len(s.strip()) > 0 ]
    return sentences

def tokenize(x, norm = True , UNIT = 'word'):
    if norm:
        x = normalize(x)
    if UNIT == "char":
        return re.sub(" ", "", x)
    if UNIT in ("word", "sent"):
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write((" ".join(seq[0]) + "\t" + " ".join(seq[1]) if seq else "") + "\n")
    fo.close()

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

def load_checkpoint(filename, model = None , ACTIVE_DEVICE = 0 , verbos = True):
    if verbos: 
        print("loading %s" % filename)
    checkpoint = torch.load(filename , map_location="cuda:" + str(ACTIVE_DEVICE))
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    #params = checkpoint["params"]
    if verbos: 
        print("saved model: epoch = %d, loss = %f" % (epoch, loss))
    return checkpoint

def save_checkpoint(filename, model, epoch, loss, time , params):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving %s" % filename)
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        if params:
            for k , v in params.items():
                checkpoint[k] = v
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def save_checkpoint_retrain(filename , model , epoch , optimizer , loss):
    print("epoch = % d , loss = %f " % (epoch , loss))
    if filename and model and optimizer:
        print("saving %s for retrain" % filename)
        checkpoint = {}
        checkpoint['epoch'] = epoch
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['loss'] = loss
        torch.save(checkpoint , filename+".pt")
        print("saved model as %s" % (filename+".pt"))

def load_checkpoint_retrain(filename , model , optimizer):
    print("loading %s" % filename)
    checkpoint = torch.load(filename)

    if model and optimizer:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("model at epoch %d loaded and can be continued for training" , checkpoint['epoch'])
        return model , optimizer, checkpoint['epoch']
    return None

class dataloader():
    def __init__(self):
        for a, b in self.data().__dict__.items():
            setattr(self, a, b)

    class data():
        def __init__(self):
            self.idx = None # input index
            self.x0 = [[]] # raw input
            self.x1 = [[]] # tokenized input
            self.xc = [[]] # indexed input, character-level
            self.xw = [[]] # indexed input, word-level
            self.yiob = [[]] # actual output
            self.yner = [[]] # actual output
            self.y0 = [[]] # actual output

            self.y1ner = [] # predicted output
            self.y1iob = [] # predicted output
            self.y1 = [] # predicted output
            self.lens = None # document lengths
            self.prob = [] # probability
            self.attn = [] # attention heatmap

    def append_item(self, x0 = None, x1 = None, xc = None, xw = None, y0 = None , yiob = None, yner = None ):
        if x0: self.x0[-1].extend(x0)
        if x1: self.x1[-1].extend(x1)
        if xc: self.xc[-1].extend(xc)
        if xw: self.xw[-1].extend(xw)
        if yiob: self.yiob[-1].extend(yiob)
        if yner: self.yner[-1].extend(yner)
        if y0: self.y0[-1].extend(y0)

    def append_row(self):
        self.x0.append([])
        self.x1.append([])
        self.xc.append([])
        self.xw.append([])
        self.yiob.append([])
        self.yner.append([])
        self.y0.append([])


    def strip(self):
        if len(self.xw[-1]):
            return
        self.x0.pop()
        self.x1.pop()
        self.xc.pop()
        self.xw.pop()
        self.y0.pop()
        self.yiob.pop()
        self.yner.pop()

    def sort(self , HRE = False):
        self.idx = list(range(len(self.x0)))
        self.idx.sort(key = lambda x: -len(self.xw[x] if HRE else self.xw[x][0]))
        self.x0 = [self.x0[i] for i in self.idx]
        self.x1 = [self.x1[i] for i in self.idx]
        self.xc = [self.xc[i] for i in self.idx]
        self.xw = [self.xw[i] for i in self.idx]

    def unsort(self):
        self.idx = sorted(range(len(self.x0)), key = lambda x: self.idx[x])
        self.x0 = [self.x0[i] for i in self.idx]
        self.x1 = [self.x1[i] for i in self.idx]
        self.xc = [self.xc[i] for i in self.idx]
        self.xw = [self.xw[i] for i in self.idx]
        if self.y1:
            self.y1 = [self.y1[i] for i in self.idx]
        if self.y1ner: 
            self.y1ner = [self.y1ner[i] for i in self.idx]
            self.y1iob = [self.y1iob[i] for i in self.idx]
        if self.prob: self.prob = [self.prob[i] for i in self.idx]
        if self.attn: self.attn = [self.attn[i] for i in self.idx]

    def split(self , BATCH_SIZE , HRE = False): # split into batches
        batch_iter = len(self.y0) if self.y0 else len(self.yiob)
        for i in range(0, batch_iter, BATCH_SIZE):
            batch = self.data()
            j = i + min(BATCH_SIZE, len(self.x0) - i)
            batch.x0 = self.x0[i:j]
            batch.yiob = self.yiob[i:j]
            batch.yner = self.yner[i:j]
            batch.y0 = self.y0[i:j]
            batch.y1 = [[] for _ in range(j - i)]
            batch.y1ner = [[] for _ in range(j - i)]
            batch.y1iob = [[] for _ in range(j - i)]
            batch.lens = [len(x) for x in self.xw[i:j]]
            batch.prob = [Tensor([0]) for _ in range(j - i)]
            batch.attn = [[] for _ in range(j - i)]
            if HRE:
                batch.x1 = [list(x) for x in self.x1[i:j] for x in x]
                batch.xc = [list(x) for x in self.xc[i:j] for x in x]
                batch.xw = [list(x) for x in self.xw[i:j] for x in x]
            else:
                batch.x1 = [list(*x) for x in self.x1[i:j]]
                batch.xc = [list(*x) for x in self.xc[i:j]]
                batch.xw = [list(*x) for x in self.xw[i:j]]
            yield batch

    def tensor(self, bc, bw, lens = None, sos = False, eos = False , HRE = False):
        _p, _s, _e = [PAD_IDX], [SOS_IDX], [EOS_IDX]
        if HRE and lens:
            d_len = max(lens) # document length (Ld)
            i, _bc, _bw = 0, [], []
            for j in lens:
                if sos:
                    _bc.append([[]])
                    _bw.append([])
                _bc.extend(bc[i:i + j] + [[[]] for _ in range(d_len - j)])
                _bw.extend(bw[i:i + j] + [[] for _ in range(d_len - j)])
                if eos:
                    _bc.append([[]])
                    _bw.append([])
                i += j
            bc, bw = _bc, _bw
        if bw:
            s_len = max(map(len, bw)) # sentence length (Ls)
            bw = [_s * sos + x + _e * eos + _p * (s_len - len(x)) for x in bw]
            bw = LongTensor(bw) # [B * Ld, Ls]
        if bc:
            w_len = max(max(map(len, x)) for x in bc) # word length (Lw)
            w_pad = [_p * (w_len + 2)]
            bc = [[_s + w + _e + _p * (w_len - len(w)) for w in x] for x in bc]
            bc = [w_pad * sos + x + w_pad * (s_len - len(x) + eos) for x in bc]
            bc = LongTensor(bc) # [B * Ld, Ls, Lw]
        return bc, bw

    def generate_output(self, sentence , output , doc_ix ):
        state = []
        ner_c = Counter()
        anotated = []
        for i,o in enumerate(output):
            o = o.strip().lower()
            if o != 'o':
                iob,ner = o.split('-')
                state.append(iob)
                if state[-1] == 'b':
                    if len(state) == 2 and state[-2] == 'b':
                        anotated[-1][1] = ner_c.most_common(1)[0][0]
                        ner_c.clear()
                        state.pop()
                    anotated.append([sentence[i] , '' , doc_ix , i])
                    ner_c.update([ner])
                elif state[-1] == 'i':
                    anotated[-1][0] += ' '+ sentence[i]
                    ner_c.update([ner])
                    state.pop()
            else:
                anotated.append([sentence[i] , 'o' , doc_ix , i])
                if state and state[-1] == 'b':
                    anotated[-2][1] = ner_c.most_common(1)[0][0]
                    ner_c.clear()
                    state.pop()
        return anotated

    def save_groupwords(self, path):
        """
        This function will group dose words need to be connected to each other and create a csv file with this structure
        -----------------------------------
        'Word' | 'NER' | 'Doc_id' | 'Word_ix'
        -----------------------------------
        Doc_id = represent the sentence index
        """
        annotations = []
        # self.x1 is the tokenize input 
        for i,s in enumerate(self.x1):
            sentence = []
            output = []
            for j,w in enumerate(s[0]):
                sentence.append(w)
                if self.y1iob[i][j].lower() == 'o':
                    output.append('o')
                else:
                    output.append(self.y1iob[i][j] + '-'+ self.y1ner[i][j])
            #print(sentence , output)
            annotated = self.generate_output(sentence , output, i )
            #print(annotated)
            #print(''.join(['-']*30))
            annotations.extend(annotated)
        #print(annotations)
        predicted = pd.DataFrame(data = annotations , columns=['Word','NER' , 'Doc_id' , 'Word_ix'] )
        predicted.to_csv(path)
    
    def save_flatwords(self, path):
        """
        This function create a csv file with this structure
        -----------------------------------
        'Word' | 'IOB' | 'NER' | 'Doc_id' | 'Word_ix'
        -----------------------------------
        Doc_id = represent the sentence index
        IOB = INSIDE OUT BEGIN
        """
        annotations = []
        # self.x1 is the tokenize input 
        for i,s in enumerate(self.x1):
            for j,w in enumerate(s[0]):
                annotations.append([w , self.y1iob[i][j] , self.y1ner[i][j] , i , j])
        predicted = pd.DataFrame(data = annotations , columns=['Word','IOB' , 'NER' , 'Doc_id' , 'Word_ix'] )
        predicted.to_csv(path)
                   
def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

def iob_to_txt(x, y): # for word/sentence segmentation
    out = [[]]
    if re.match("(\S+/\S+( |$))+", x): # token/tag
        x = re.sub(r"/[^ /]+\b", "", x) # remove tags
    for i, (j, k) in enumerate(zip(tokenize(x, False), y)):
        if i and k[0] == "B":
            out.append([])
        out[-1].append(j)
    if TASK == "word-segmentation":
        d1, d2 = "", " "
    if TASK == "sentence-segmentation":
        d1, d2 = " ", "\n"
    return d2.join(d1.join(x) for x in out)

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0

def metrics(actual , pred , model_name = None , save = False , filename = None):
    """
    actula : Actual output
    pred : Predicted output
    model_name : The model that is evaluated to be saved
    save : create a csv to save the output
    filename : to save the result we need a file name if this file is exists we will append the new result
    NOTE : To see more results, save the file we will present more details
    """
    assert len(actual) == len(pred) , "Actual output ( "+str(len(actual)) + ") and Predicted out put ("+str(len(pred))+") are diffrent in length! "
    print_result = {'macro_precision' : 0.0,'macro_recall' : 0.0,'hmacro_f1': 0.0 , 'amacro_f1': 0.0 ,'micro_f1': 0.0,'auc': 0.0 }
    columns = ['Model','label','precision','recall','f1' ,'TP','SUPPORT',
    'macro_precision','macro_recall','hmacro_f1' , 'amacro_f1' ,'micro_f1','auc']
    avg = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    output_values = {}
    y_test , y_pred = [] , []
    data = []
    for  act, prd in zip(actual , pred): # actual value, prediction
        assert len(act) == len(prd), "Actual output ( "+ str(len(act)) + ") and Predicted out put ("+str(len(prd))+") are diffrent in length! "
        for a, p in zip(act, prd):
            y0 = str(a).strip().lower()
            y1 = str(p).strip().lower()
            if y0 not in output_values:
                output_values[y0] = len(output_values) + 1
            if y1 not in output_values:
                output_values[y1] = len(output_values) + 1
            y_test.append(output_values[y0])
            y_pred.append(output_values[y1])
            tp[y0] += (y0 == y1)
            tpfn[y0] += 1
            tpfp[y1] += 1
    for y in sorted(tpfn.keys()):
        pr = (tp[y] / tpfp[y]) if tpfp[y] else 0
        rc = (tp[y] / tpfn[y]) if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        avg["amacro_f1"] += f1(pr, rc)
        row = ['']
        #if not summary:
        #print("label = %s" % y)
        row.append(y)
        #print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
        row.append(pr)
        #print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
        row.append(rc)
        #print("f1 = %f\n" % f1(pr, rc))
        row.append(f1(pr, rc))
        row.append(tp[y])
        row.append(tpfn[y])
        row.extend([0,0,0,0,0,0])
        data.append(row)
    if len(tpfn) != 0:
        avg["macro_pr"] /= len(tpfn)
    else:
        avg["macro_pr"] = 0
    print_result["macro_pr"] = avg["macro_pr"]

    if len(tpfn) != 0:
        avg["macro_rc"] /= len(tpfn)
    else:
        avg["macro_rc"] = 0

    print_result["macro_rc"] = avg["macro_rc"]
    avg["micro_f1"] = sum(tp.values()) / sum(tpfn.values()) if sum(tpfn.values()) != 0 else 0
    print_result["micro_f1"] = avg["micro_f1"]

    # |F1 = so benivelant and good
    avg["hmacro_f1"] = f1(avg["macro_pr"], avg["macro_rc"])
    print_result["hmacro_f1"] = avg["hmacro_f1"]
    # F1 = arithmatic f1 score
    avg["amacro_f1"] /= len(tpfn) if len(tpfn) != 0 else 0
    print_result["amacro_f1"] = avg["amacro_f1"]

    try:
        fpr , tpr , _  = roc_curve(y_test , y_pred , pos_label=len(output_values))
        AUC =  auc(fpr , tpr)
        print_result["auc"] = AUC
    except:
        AUC =  0
        print_result["auc"] = AUC
    
    #print("AUC:" , auc(fpr , tpr))
    data.append([model_name ,'Total',0,0,0,0,0, avg["macro_pr"] , avg["macro_rc"] , avg["hmacro_f1"] , avg["amacro_f1"] , avg["micro_f1"], AUC])
    results = None
    if save:
        results = pd.DataFrame(columns=columns , data = data)
        old_res = None
        if isfile(filename+'.csv'):
            old_res = pd.read_csv(filename + '.csv')
        if old_res is not None:
            results = pd.concat([old_res , results]) 
        results.to_csv(filename +'.csv' , index=False)
    return print_result

def evaluation_report(parameter , output_filename = 'bestmodel'):
    """
    parameter = {
    'metric' : 'amacro_f1',
    'label' : 'Total',
    'dbname' : '_NCBI-disease-IOB'
        }
    """
    best_results = []
    for d in os.listdir():
        if d.startswith('rnn'):
            for r in os.listdir(os.path.join(d)):
                if r.endswith(".csv"):
                    embed_type = re.sub('rnn_two_crf_|rnn_single_crf_|rnn_two_crf_seq_|rnn_two_crf_seq2_|rnn_two_crf_par_' , '' , d.replace(parameter['dbname'] , '')) 
                    res = pd.read_csv(os.path.join(d,r))
                    res = res[res['label'] == parameter['label']][['Model' , parameter['metric']]]
                    res['epoch'] = res['Model'].apply(lambda x : re.findall('_\d+',x)[0].replace('_' , '') if re.findall('_\d+',x) else 0)
                    res['Model'] = res['Model'].apply(lambda x : embed_type+'_'+re.sub('_\d+','', x ))
                    res_max =  res.groupby(['Model']).max().reset_index()
                    best_results.append(res_max)
    results = pd.concat(best_results).reindex()
    results.to_csv( output_filename + '.csv')