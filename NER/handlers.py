from utils import * 
from model import *
import re
import os

class Handler():
    def __init__(self ):
        self.cti = None
        self.wti = None
        self.tti_iob = None
        self.tti_ner = None
        self.tti = None
        self.itt = None
        self.itt_iob = None
        self.itt_ner = None
        self.itw = None
        self.itc = None
        self.data = None
        self.batch_handler = None
        self.data_handler = dataloader()
        self.BATCH_SIZE = None
        self.HRE = False

    def readtsv(self, filename):
        """
        data will be return as a tupel (word,tag) 
        it will be suitable for load_data function
        data =[ [(word , tag) , ...]]
        """
        data = []
        with open(filename) as f:
            sentence = []
            for l in f:
                l = l.lower().strip()
                if l and  not l.startswith('-docstart-'):
                    sentence.append(re.split('\s+' , l))
                else:
                    if sentence:
                        # it means we have a sentence contains just one word
                        if len(sentence) == 1:
                            # add a dot to be finish the sentence
                            sentence.append(['.' , 'o'])
                        data.append(sentence.copy())
                        sentence.clear()
        return data

    def load_data(self, cti , wti , tti_iob, tti_ner , tti ,sentences , BATCH_SIZE , HRE = False , load_percentage = 1):
        data = dataloader()
        batch = []
        block = []
        # total sentences is loaded
        tsl = 0
        for si, s in enumerate(sentences):
            loaded_percentage = (si+1) / len(sentences)
            if loaded_percentage > load_percentage:
                print('%.2f is loaded / %d'%(load_percentage , len(sentences)))
                break
            xy = []
            for w,t in s:
                w = w.lower()
                t = t.lower()
                wxc = [cti[c] for c in w]
                # characters , word , tag
                if '-' in t:
                    iob , ner = t.split('-')
                    xy.append((wxc , wti[w] , tti_iob[iob] , tti_ner[ner] , tti[t]))
                else:    
                    xy.append((wxc , wti[w] , tti_iob[t] , tti_ner[t] , tti[t]))
            # * it will be used to unzip the list
            xc , xw , yiob , yner , y0 = zip(*xy)
            sl = len(s)
            block.append((sl , xc , xw , yiob , yner , y0))
            tsl = si+1
        # sort based on the longest sentences (sequence)
        block.sort(key=lambda x: -x[0])
        for s in block:
            data.append_item(xc = [list(s[1])] , xw = [list(s[2])] , yiob = s[3] , yner = s[4] , y0= s[5])
            data.append_row()
        data.strip()
        for _batch in data.split(BATCH_SIZE , HRE):
            xc, xw = data.tensor(_batch.xc, _batch.xw, _batch.lens)
            _, yiob = data.tensor(None, _batch.yiob, sos = True)
            _, yner = data.tensor(None, _batch.yner, sos = True)
            _, y0 = data.tensor(None, _batch.y0, sos = True)
            batch.append((xc, xw, yiob , yner , y0))
        print('%d/%d sentenced is loaded'%(tsl,len(sentences)))
        self.batch_handler = batch 
        self.data_handler = data

class ModelHandler(Handler):

    def __init__(self , MODE , params , db_location =  "datasets/JNLPBA/",  sample_test = False , load_percentage = 1):
        
        super().__init__()
        self.MODE = MODE
        if MODE != 'load':
            params['EMBED_SIZE'] = sum(params['EMBED'].values())
            params['HRE'] = (params['UNIT'] == "sent")
            self.BATCH_SIZE = params["BATCH_SIZE"]
            self.HRE = params["HRE"]
            self.params = params
            self.JNLPBA_LOCATION = db_location
            if sample_test:
                self.data = [
                    [('IL-2','B-DNA'),('gene','I-DNA'),('expression','O'),('and','O'),('NF-kappa','B-protein'),('B','I-protein'),('activation','O'),('through','O'),('CD28','B-protein'),('requires','O'),('reactive','O'),('oxygen','O'),('production','O'),('by','O'),('5-lipoxygenase','B-protein'),('.','O')],
                    [('Activation','O'),('of','O'),('the','O'),('CD28','B-protein'),('surface','I-protein'),('receptor','I-protein'),('provides','O'),('a','O'),('major','O'),('costimulatory','O'),('signal','O'),('for','O'),('T','O'),('cell','O'),('activation','O'),('resulting','O'),('in','O'),('enhanced','O'),('production','O'),('of','O'),('interleukin-2','B-protein'),('(','O'),('IL-2','B-protein'),('),','O'),('and','O'),('cell','O'),('proliferation','O'),('.','O')]
                    ]
                self.devel_sentences = ["Number of glucocorticoid receptors in lymphocytes and their sensitivity to hormone action ." , 
                "The study demonstrated a decreased level of glucocorticoid receptors ( GR ) in peripheral blood lymphocytes from hypercholesterolemic subjects , and an elevated level in patients with acute myocardial infarction ."]
                self.devel_target_iob = [
                                    ['O' ,'O' ,'B' ,'I' ,'O' ,'B' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O'],
                                    ['O','O','O','O','O','O','O','B','I','O','B','O','O','B','I','I','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O']
                                ]
                self.devel_target_ner = [
                                    ['O' ,'O' ,'protein' ,'protein' ,'O' ,'cell_type' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O'],
                                    ['O','O','O','O','O','O','O','protein','protein','O','protein','O','O','cell_type','cell_type','cell_type','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O']
                                ]
                self.devel_target = [
                                    ['O' ,'O' ,'b-protein' ,'i-protein' ,'O' ,'b-cell_type' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O' ,'O'],
                                    ['O','O','O','O','O','O','O','b-protein','i-protein','O','b-protein','O','O','b-cell_tyoe','i-cell_type','i-cell_type','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O']
                                ]
            else:                        
                self.data = self.readtsv(self.JNLPBA_LOCATION + MODE +".tsv")
                
            self.cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
            self.wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX}
            self.tti_iob = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX , O:O_IDX, B:B_IDX  , I:I_IDX }
            self.tti_ner = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, O:O_IDX}
            self.tti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX}
            for s in self.data:
                for w , t in s:
                    w = w.lower()
                    t = t.lower()
                    if w not in self.wti:
                        self.wti[w] = len(self.wti)
                    if t not in self.tti:
                        self.tti[t] = len(self.tti)
                    if '-' in t:
                        iob, ner = t.split('-')
                        if ner not in self.tti_ner:
                            self.tti_ner[ner] = len(self.tti_ner)
                        if iob not in self.tti_iob:
                            self.tti_iob[iob] = len(self.tti_iob)
                    for c in w:
                        if c not in self.cti:
                            self.cti[c] = len(self.cti)

            #save_tkn_to_idx(self.JNLPBA_LOCATION +self.MODE+".wti" , self.wti)
            #save_tkn_to_idx(self.JNLPBA_LOCATION +self.MODE+".cti" , self.cti)
            #save_tkn_to_idx(self.JNLPBA_LOCATION +self.MODE+".tti_iob" , self.tti_iob)
            #save_tkn_to_idx(self.JNLPBA_LOCATION +self.MODE+".tti_ner" , self.tti_ner)
            #save_tkn_to_idx(self.JNLPBA_LOCATION +self.MODE+".tti" , self.tti)

            self.itt = {v:k for k,v in self.tti.items()}
            self.itt_iob = {v:k for k,v in self.tti_iob.items()}
            self.itt_ner = {v:k for k,v in self.tti_ner.items()}
            self.itw = {v:k for k,v in self.wti.items()}
            self.itc = {v:k for k,v in self.cti.items()}
            self.devel_sentences = []
            self.devel_target_ner = []
            self.devel_target_iob = []
            self.devel_target = []
            self.load_data(self.cti,self.wti,self.tti_iob , self.tti_ner , self.tti, self.data , self.BATCH_SIZE,self.HRE , load_percentage= load_percentage)
            print('TRAIN Data is loaded.')
            EVAL_EVERY = self.params["EVAL_EVERY"]
            if EVAL_EVERY and not sample_test:
                assert isfile(self.JNLPBA_LOCATION + 'devel.tsv') , 'devel.tsv is not avaiable in %s'%self.JNLPBA_LOCATION 
                deval_data = self.readtsv(self.JNLPBA_LOCATION + 'devel.tsv')
                print('DEVEL Data is loaded.')
                for s in deval_data:
                    self.devel_sentences.append(' '.join([w.lower() for w,t in s]).strip())
                    self.devel_target.append([t for w ,t in s])
                    self.devel_target_ner.append([ re.sub('b-|i-', '' , t.strip().lower()) for w,t in s])
                    self.devel_target_iob.append([ re.split('-', t.strip().lower())[0] for w,t in s])

    def train(self , output_path = '' , retrain = False , model_path = ''):
        assert self.MODE.lower().strip() == 'train' , "To train please make sure you have your MODE = train and also have the train.tsv avaiable in the dataset/JNLPBA directory"
        model = None
        LEARNING_RATE = self.params["LEARNING_RATE"]
        num_epoch = self.params["EPOCH"]
        SAVE_EVERY = self.params["SAVE_EVERY"]
        model_name = self.params["model_name"]
        EVAL_EVERY = self.params["EVAL_EVERY"]
        model = None
        init_epoch = 1
        if retrain:
            print(''.join(['=']*20) + 'Continue Training' + ''.join(['=']*20))
            model = self.load_model(model_path , ACTIVE_DEVICE)
            init_epoch = int(re.findall('epoch\d+' , model_path.lower())[0].replace('epoch','')) + 1
            print('Continue at ' + str(init_epoch) )
        else:
            assert model_name.strip() ,  "model name is empty, choose one of the available model rnn_two_crf_par , rnn_two_crf, rnn_two_crf_seq, rnn_single_crf"
            if model_name.lower() == 'rnn_two_crf_par':
                model = rnn_two_crf_par(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
            elif model_name.lower() == 'rnn_two_crf':
                model = rnn_two_crf(len(self.cti), len(self.wti), max(len(self.tti_iob) , len(self.tti_ner)) , len(self.tti_iob) , len(self.tti_ner) , self.params)
            elif model_name.lower() == 'rnn_two_crf_seq':
                model = rnn_two_crf_seq(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
            elif model_name.lower() == 'rnn_two_crf_seq2':
                model = rnn_two_crf_seq2(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
            elif model_name.lower() == 'rnn_single_crf':
                model = rnn_single_crf(len(self.cti), len(self.wti) , len(self.tti) , self.params)
            print(model)
        
        optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        print("Adam Optimizer is using with learning rate of : %.4f"%LEARNING_RATE)
        print("training %s model..."%model_name)
        for e in range(init_epoch , num_epoch+1):
            loss_sum_iob , loss_sum_ner = 0 , 0
            loss_sum = 0
            timer = time()
            for xc, xw, yiob, yner, y0 in self.batch_handler:
                if model_name.lower() == 'rnn_two_crf_par' or model_name.lower() == 'rnn_two_crf':
                    loss = model(xc,xw,yiob,yner)
                    loss.backward()
                    loss_sum +=loss.item()
                elif model_name == 'rnn_single_crf':
                    loss = model(xc,xw,y0)
                    loss.backward()
                    loss_sum +=loss.item()
                elif model_name == 'rnn_two_crf_seq' or model_name == 'rnn_two_crf_seq2':
                    loss_iob , loss_ner = model(xc,xw,yiob , yner)
                    loss_iob.backward(retain_graph=True)
                    loss_ner.backward()
                    loss_sum_iob += loss_iob.item()
                    loss_sum_ner += loss_ner.item()
                optim.step()
            timer = time() - timer
            if model_name == 'rnn_two_crf_seq' or model_name == 'rnn_two_crf_seq2':
                loss_sum_iob /= len(self.batch_handler)
                loss_sum_ner /= len(self.batch_handler)
                print('loss_iob :%.2f  loss_ner:%.2f'%(loss_iob,loss_ner))
                loss_sum = (loss_sum_iob + loss_sum_ner)/2
            else:
                loss_sum /= len(self.batch_handler)

            if e % SAVE_EVERY and e != num_epoch:
                save_checkpoint("", None, e, loss_sum, timer, None)
            else:
                if output_path and not os.path.isdir(output_path):
                    os.mkdir(output_path)
                save_checkpoint(os.path.join(output_path , model_name) , model, e, loss_sum, timer , {"params": self.params , "cti" : self.cti , "wti" : self.wti , "tti" : self.tti , "tti_iob": self.tti_iob , "tti_ner": self.tti_ner })
            if EVAL_EVERY and (e % EVAL_EVERY == 0 or e == num_epoch):
                if model_name == '':
                    model.evaluate(self.devel_sentences, self.cti, self.wti, self.itt, 
                           y0 = self.devel_target, 
                           parameters =['amacro_f1'],
                           model_name =model_name, 
                           save = True ,
                           filename = model_name)
                else:
                    if output_path and not os.path.isdir(output_path):
                        os.mkdir(output_path)
                    if model_name == 'rnn_single_crf':
                        model.evaluate(self.devel_sentences , self.cti , self.wti , self.itt,
                        self.devel_target, parameters=['amacro_f1'] , model_name=model_name + '_' + str(e) , save=True , filename= os.path.join(output_path , model_name) )
                    else:
                        model.evaluate(self.devel_sentences , self.cti , self.wti , self.itt_iob , self.itt_ner , 
                        self.devel_target_iob , self.devel_target_ner , parameters=['amacro_f1'] , model_name=model_name + '_' + str(e) , save=True , filename= os.path.join(output_path , model_name) )

            print()
    
    def retrain(self, model_path , init_epoch , map_location = 0):
        model = self.load_model(model_path , map_location)
        LEARNING_RATE = self.params["LEARNING_RATE"]
        num_epoch = self.params["EPOCH"]
        SAVE_EVERY = self.params["SAVE_EVERY"]
        model_name = self.params["model_name"]
        EVAL_EVERY = self.params["EVAL_EVERY"]
        optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    def load_model(self, model_path , map_location = 0):
        checkpoint = load_checkpoint(filename = model_path , ACTIVE_DEVICE=map_location)
        print(checkpoint["params"])
        model_name = checkpoint["params"]["model_name"]
        self.params = checkpoint["params"]
        self.cti = checkpoint["cti"]
        self.wti = checkpoint["wti"]
        self.tti_iob = checkpoint["tti_iob"]
        self.tti_ner = checkpoint["tti_ner"]
        self.tti = checkpoint["tti"]
        self.itt = {v:k for k,v in self.tti.items()}
        self.itt_iob = {v:k for k,v in self.tti_iob.items()}
        self.itt_ner = {v:k for k,v in self.tti_ner.items()}
        self.itw = {v:k for k,v in self.wti.items()}
        self.itc = {v:k for k,v in self.cti.items()}
        if model_name.lower() == 'rnn_two_crf_par':
            model = rnn_two_crf_par(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
        elif model_name.lower() == 'rnn_two_crf':
            model = rnn_two_crf(len(self.cti), len(self.wti), max(len(self.tti_iob) , len(self.tti_ner)) , len(self.tti_iob) , len(self.tti_ner) , self.params)
        elif model_name.lower() == 'rnn_two_crf_seq':
            model = rnn_two_crf_seq(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
        elif model_name.lower() == 'rnn_two_crf_seq2':
            model = rnn_two_crf_seq2(len(self.cti), len(self.wti), len(self.tti_iob) , len(self.tti_ner) , self.params)
        elif model_name.lower() == 'rnn_single_crf':
            model = rnn_single_crf(len(self.cti), len(self.wti) , len(self.tti) , self.params)
        print(model)

        if model:
            model.load_state_dict(checkpoint["state_dict"])

        return model
    
    def load_model_test(self, model_path, test_data_location , result_location , map_location = 0):
        """
        model_path = location of model exp: ../models/model.ckp
            No restiction on extention of the file (ckp = check point)
        test_data_location = location of test data this must be a Tab Seperated Value (.tsv) file
        result_location = is the directory that we want to save the result of the rest this directory if is not exist will be generated
        map_location = is an integer that let to assign the model to any gpu when you have multiple gpu (default is 0)
        """
        model = self.load_model(model_path, map_location)
        test_data = self.readtsv(test_data_location)
        test_sentences = []
        test_target_ner = []
        test_target_iob = []
        test_target = []
        for s in test_data:
            test_sentences.append(' '.join([w.lower() for w,t in s]).strip())
            test_target.append([t for w ,t in s])
            test_target_ner.append([ re.sub('b-|i-', '' , t.strip().lower()) for w,t in s])
            test_target_iob.append([ re.split('-', t.strip().lower())[0] for w,t in s])

        output_path = result_location
        if output_path and not os.path.isdir(output_path):
            os.mkdir(output_path)
        embed = list(self.params["EMBED"].keys())[0]
        if self.params["model_name"] == 'rnn_single_crf':
            model.evaluate(test_sentences , self.cti , self.wti , self.itt , 
                test_target , parameters=['amacro_f1'] , model_name=self.params["model_name"]+"_"+embed , save=True , filename= os.path.join(output_path , self.params["model_name"]) )
        else:
            model.evaluate(test_sentences , self.cti , self.wti , self.itt_iob , self.itt_ner , 
                test_target_iob , test_target_ner , parameters=['amacro_f1'] , model_name=self.params["model_name"]+"_"+embed , save=True , filename= os.path.join(output_path , self.params["model_name"]) )
        print()
        del model
        torch.cuda.empty_cache()