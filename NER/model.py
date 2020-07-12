from NER.utils import *
from NER.embedding import embed
import torch
import matplotlib.pyplot as plt


# single Task
class rnn_single_crf(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags , params):
        super().__init__()
        self.rnn = rnn(cti_size, wti_size, num_tags , params)
        self.crf = crf(num_tags , params)
        self = self.cuda(ACTIVE_DEVICE) if CUDA else self
        self.HRE = params['HRE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.params = params

    def forward(self, xc, xw, y0): # for training
        self.zero_grad()
        self.rnn.batch_size = y0.size(0)
        self.crf.batch_size = y0.size(0)
        mask = y0[:, 1:].gt(PAD_IDX).float()
        #print("xw", xw.shape)
        #print('mask' , mask.shape)
        h = self.rnn(xc, xw, mask)
        #print("h :" , h.shape)
        Z = self.crf.forward(h, mask)
        #print("Y0 :" , y0 , Z)
        score = self.crf.score(h, y0, mask)
        #print("score :", score)
        return torch.mean(Z - score) # NLL loss

    def decode(self, xc, xw, doc_lens): # for inference
        self.rnn.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crf.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        if self.HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (doc_lens[0] - x) for x in doc_lens])
        else:
            mask = xw.gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        return self.crf.decode(h, mask)

    def loaddata(self, sentences, cti, wti, itt , y0 = None):
        data = dataloader()
        block = []
        for si, sent in enumerate(sentences):
            sent = normalize(sent)
            words = tokenize(sent) #sent.split(' ')
            x = []
            for w in words:
                w = normalize(w)
                wxc = [cti[c] if c in cti else UNK_IDX for c in w]
                x.append((wxc , wti[w] if w in wti else UNK_IDX))
            xc , xw = zip(*x)
            if y0:
                assert len(y0[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                block.append((sent, xc,xw , y0[si]))    
            else:
                block.append((sent, xc,xw))
        for s in block:
            if y0:
                data.append_item( x0 = [s[0]] , xc= [list(s[1])] , xw =[list(s[2])] , y0 = s[3])
                data.append_row()
            else:
                data.append_item( x0 = [s[0]] , xc= [list(s[1])] , xw =[list(s[2])] , y0 = [])
                data.append_row()
        data.strip()
        data.sort()
        return data

    def predict(self , sentences , cti , wti , itt ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        """
        data = self.loaddata(sentences,cti,wti,itt )
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1 = self.decode(xc, xw, batch.lens)
            data.y1.extend([[itt[i]  for i in x] for x in y1])
        data.unsort()
        return data

    def evaluate(self, sentences, cti , wti , itt , y0 , parameters = [] ,  model_name = None , save = False , filename = None ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        y0 : Target values to evaluate
        parameters : 'macro_precision','macro_recall','hmacro_f1', 'amacro_f1','micro_f1','auc'
        """
        data = self.loaddata(sentences, cti, wti , itt , y0)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1 = self.decode(xc, xw, batch.lens)
            data.y1.extend([[itt[i]  for i in x] for x in y1])
        data.unsort()
        result = metrics(data.y0 , data.y1 , model_name=model_name,save=save,filename=filename)
        if parameters:
            print("============ evaluation results ============") 
            for m in parameters:
                if m in result:
                    print("\t" + m +" = %f"% result[m])
        return data

# Two Sequential
# this is tested based on jupyter Run-rnn_two_crf_seq
class rnn_two_crf_seq(nn.Module):
    def __init__(self, cti_size, wti_size , num_tags_iob , num_tag_ner  , params):
        """
        num_output_rnn : Maximum number of out put for the two iob and ner
        """
        super().__init__()
        self.rnn_iob = rnn(cti_size, wti_size, num_tags_iob , params)
        self.crfiob = crf(num_tags_iob , params)
        
        self.HRE = params['HRE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.NUM_DIRS = params['NUM_DIRS']
        self.NUM_LAYERS = params['NUM_LAYERS']
        self.DROPOUT = params['DROPOUT']
        self.RNN_TYPE = params['RNN_TYPE']
        self.HIDDEN_SIZE = params['HIDDEN_SIZE']
        self.rnn_ner = getattr(nn, self.RNN_TYPE)(
            input_size = num_tags_iob,
            hidden_size = self.HIDDEN_SIZE // self.NUM_DIRS,
            num_layers = self.NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = self.DROPOUT,
            bidirectional = (self.NUM_DIRS == 2)
        )
        self.out = nn.Linear(self.HIDDEN_SIZE, num_tag_ner) # RNN output to tag
        self.crfner = crf(num_tag_ner , params)
        self.params = params
        self = self.cuda(ACTIVE_DEVICE) if CUDA else self

    def forward(self, xc, xw, yiob , yner): # for training
        self.zero_grad()
        self.rnn_iob.batch_size = xw.size(0)
        self.rnn_ner.batch_size = xw.size(0)
        self.crfiob.batch_size = xw.size(0)
        self.crfner.batch_size = xw.size(0)

        # Mask on sentence
        mask = xw[:,1:].gt(PAD_IDX).float()
        
        # Layer one get the embed and then go to crf for IOB
        h_iob = self.rnn_iob(xc, xw, mask)
        #mask_iob = yiob[:, 1:].gt(PAD_IDX).float()
        #print('MASK_IOB')
        #print(mask_iob)
        # this need to be backward when we train it ()
        Ziob = self.crfiob.forward(h_iob, mask)
        # Result of IOB will go to the RNN and Predict the NER
        h_iob *= mask.unsqueeze(2)
            
        h_ner , _ = self.rnn_ner(h_iob)
        ner_out = self.out(h_ner)
        t = 2
        # to see how CRF converge the model to the output
        #for _nerpred, _nery in zip(ner_out , yner):
        #    plt.plot(_nerpred[_nery[t+1]].cpu().data.numpy())
        #    plt.vlines(x=_nery[t+1].cpu().data.numpy(),ymin= 0 , ymax=1)
        #    plt.show()
            #print(_nerpred[_nery[t+1]].data , _nery[t+1].data)
        #ner_out *= mask.unsqueeze(2)
        #mask_ner = yner[:, 1:].gt(PAD_IDX).float()
        #print('MASK_NER')
        #print(mask_ner)
        Zner = self.crfner.forward(ner_out, mask)

        scoreiob = self.crfiob.score(h_iob, yiob, mask)
        scorener = self.crfner.score(ner_out, yner, mask)

        return  torch.mean(Ziob - scoreiob) , torch.mean(Zner - scorener) # NLL loss

    def decode(self, xc, xw, doc_lens): # for inference
        self.rnn_iob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.rnn_ner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfiob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        if self.HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (doc_lens[0] - x) for x in doc_lens])
        else:
            mask = xw.gt(PAD_IDX).float()

        iob_pred = self.rnn_iob(xc, xw, mask)
        #iob_pred = self.iob(h_bio)

        h_ner , _ = self.rnn_ner(iob_pred)
        

        ner_pred = self.out(h_ner)

        return self.crfiob.decode(iob_pred, mask) , self.crfner.decode(ner_pred, mask)

    def loaddata(self, sentences , cti , wti , itt_iob , itt_ner , yiob=None , yner =None):
        data = dataloader()
        block = []
        for si ,sent in enumerate(sentences) :
            sent = normalize(sent)
            words = tokenize(sent) #sent.split(' ')
            x = []
            tokens = []
            for w in words:
                w = normalize(w)
                wxc = [cti[c] if c in cti else UNK_IDX for c in w]
                x.append((wxc , wti[w] if w in wti else UNK_IDX))
                tokens.append(w)
            xc , xw = zip(*x)
            if yiob and yner:
                assert len(yiob[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                assert len(yner[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                block.append((sent,tokens, xc,xw , yiob[si] , yner[si]))    
            else:
                block.append((sent,tokens, xc,xw))
        for s in block:
            if yiob and yner:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=s[4] , yner =s[5])
                data.append_row()
            else:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=[]   , yner =[]  )
                data.append_row()
        data.strip()
        data.sort()
        return data

    def predict(self , sentences , cti , wti , itt_iob , itt_ner):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt_iob : Index To Tag  IOB (Inside Other Begin)
        itt_ner : Index To Tag Named Entity REcognition
        """
        #itt_iob = {v:k for k,v in tti_iob.items()}
        #itt_ner = {v:k for k,v in tti_ner.items()}
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            yiob , yner = self.decode(xc, xw, batch.lens)
            #print(yiob , yner)
            #print([[itt_iob[i] if i in itt_iob else O for i in x] for x in yiob])
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in yiob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in yner])
        data.unsort()
        #print(data.y1iob , data.x1)
        return data

    def evaluate(self, sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner , parameters = [] ,  model_name = None , save = False , filename = None ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        y0iob : Target values to evaluate (Inside Other Begin)
        y0ner : Target values to evaluate (Named Entity Recognition)
        parameters : 'macro_precision','macro_recall','hmacro_f1', 'amacro_f1','micro_f1','auc'
        """
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1iob , y1ner = self.decode(xc, xw, batch.lens)
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in y1iob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in y1ner])
        data.unsort()
        result_ner = metrics(data.yner , data.y1ner , model_name=model_name + "_ner",save=save,filename=filename)
        result_iob = metrics(data.yiob , data.y1iob , model_name=model_name + "_iob",save=save,filename=filename)
        if parameters:
            print("============ evaluation results ============") 
            for m in parameters:
                if m in result_ner:
                    print("\t" + m +" (ner) = %f"% result_ner[m])
                if m in result_iob:
                    print("\t" + m +" (iob) = %f"% result_iob[m])
        return data

class rnn_two_crf_seq2(nn.Module):
    def __init__(self, cti_size, wti_size , num_tags_iob , num_tags_ner  , params):
        """
        num_output_rnn : Maximum number of out put for the two iob and ner
        """
        super().__init__()
        self.rnn_ner = rnn(cti_size, wti_size, num_tags_ner , params)
        self.crfner = crf(num_tags_ner , params)
        
        self.HRE = params['HRE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.NUM_DIRS = params['NUM_DIRS']
        self.NUM_LAYERS = params['NUM_LAYERS']
        self.DROPOUT = params['DROPOUT']
        self.RNN_TYPE = params['RNN_TYPE']
        self.HIDDEN_SIZE = params['HIDDEN_SIZE']
        self.rnn_iob = getattr(nn, self.RNN_TYPE)(
            input_size = num_tags_ner,
            hidden_size = self.HIDDEN_SIZE // self.NUM_DIRS,
            num_layers = self.NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = self.DROPOUT,
            bidirectional = (self.NUM_DIRS == 2)
        )
        self.out = nn.Linear(self.HIDDEN_SIZE, num_tags_iob) # RNN output to tag
        self.crfiob = crf(num_tags_iob , params)
        self.params = params
        self = self.cuda(ACTIVE_DEVICE) if CUDA else self

    def forward(self, xc, xw, yiob , yner): # for training
        self.zero_grad()
        self.rnn_iob.batch_size = xw.size(0)
        self.rnn_ner.batch_size = xw.size(0)
        self.crfiob.batch_size = xw.size(0)
        self.crfner.batch_size = xw.size(0)

        # Mask on sentence
        mask = xw[:,1:].gt(PAD_IDX).float()
        
        # Layer one get the embed and then go to crf for IOB
        h_ner = self.rnn_ner(xc, xw, mask)
        #mask_iob = yiob[:, 1:].gt(PAD_IDX).float()
        #print('MASK_IOB')
        #print(mask_iob)
        # this need to be backward when we train it ()
        Zner = self.crfner.forward(h_ner, mask)
        # Result of IOB will go to the RNN and Predict the NER
        h_ner *= mask.unsqueeze(2)
            
        h_iob , _ = self.rnn_iob(h_ner)
        iob_out = self.out(h_iob)
        #t = 2
        # to see how CRF converge the model to the output
        # for _nerpred, _nery in zip(ner_out , yner):
        #     plt.plot(_nerpred[_nery[t+1]].cpu().data.numpy())
        #     plt.vlines(x=_nery[t+1].cpu().data.numpy(),ymin= 0 , ymax=1)
        #     plt.show()
            #print(_nerpred[_nery[t+1]].data , _nery[t+1].data)
        #ner_out *= mask.unsqueeze(2)
        #mask_ner = yner[:, 1:].gt(PAD_IDX).float()
        #print('MASK_NER')
        #print(mask_ner)
        Ziob = self.crfiob.forward(iob_out, mask)

        scorener = self.crfner.score(h_ner, yner, mask)
        scoreiob = self.crfiob.score(iob_out, yiob, mask)

        return  torch.mean(Ziob - scoreiob) , torch.mean(Zner - scorener) # NLL loss

    def decode(self, xc, xw, doc_lens): # for inference
        self.rnn_iob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.rnn_ner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfiob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        if self.HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (doc_lens[0] - x) for x in doc_lens])
        else:
            mask = xw.gt(PAD_IDX).float()

        ner_pred = self.rnn_ner(xc, xw, mask)
        #iob_pred = self.iob(h_bio)

        h_iob , _ = self.rnn_iob(ner_pred)
        

        iob_pred = self.out(h_iob)

        return self.crfiob.decode(iob_pred, mask) , self.crfner.decode(ner_pred, mask)
    
    def loaddata(self, sentences , cti , wti , itt_iob , itt_ner , yiob=None , yner =None):
        data = dataloader()
        block = []
        for si ,sent in enumerate(sentences) :
            sent = normalize(sent)
            words = tokenize(sent) #sent.split(' ')
            x = []
            tokens = []
            for w in words:
                w = normalize(w)
                wxc = [cti[c] if c in cti else UNK_IDX for c in w]
                x.append((wxc , wti[w] if w in wti else UNK_IDX))
                tokens.append(w)
            xc , xw = zip(*x)
            if yiob and yner:
                assert len(yiob[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                assert len(yner[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                block.append((sent, tokens, xc,xw , yiob[si] , yner[si] ))    
            else:
                block.append((sent, tokens, xc,xw))
        for s in block:
            if yiob and yner:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=s[4] , yner =s[5])
                data.append_row()
            else:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=[]   , yner =[]  )
                data.append_row()
        data.strip()
        data.sort()
        return data

    def predict(self , sentences , cti , wti , itt_iob , itt_ner):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt_iob : Index To Tag  IOB (Inside Other Begin)
        itt_ner : Index To Tag Named Entity REcognition
        """
        #itt_iob = {v:k for k,v in tti_iob.items()}
        #itt_ner = {v:k for k,v in tti_ner.items()}
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            yiob , yner = self.decode(xc, xw, batch.lens)
            #print(yiob , yner)
            #print([[itt_iob[i] if i in itt_iob else O for i in x] for x in yiob])
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in yiob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in yner])
        data.unsort()
        return data

    def evaluate(self, sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner , parameters = [] ,  model_name = None , save = False , filename = None ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        y0iob : Target values to evaluate (Inside Other Begin)
        y0ner : Target values to evaluate (Named Entity Recognition)
        parameters : 'macro_precision','macro_recall','hmacro_f1', 'amacro_f1','micro_f1','auc'
        """
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1iob , y1ner = self.decode(xc, xw, batch.lens)
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in y1iob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in y1ner])
        data.unsort()
        result_ner = metrics(data.yner , data.y1ner , model_name=model_name + "_ner",save=save,filename=filename)
        result_iob = metrics(data.yiob , data.y1iob , model_name=model_name + "_iob",save=save,filename=filename)
        if parameters:
            print("============ evaluation results ============") 
            for m in parameters:
                if m in result_ner:
                    print("\t" + m +" (ner) = %f"% result_ner[m])
                if m in result_iob:
                    print("\t" + m +" (iob) = %f"% result_iob[m])
        return data

# Two Task Prallel
# This is tested based on Jupyter Run-rnn_two_crf_par
class rnn_two_crf_par(nn.Module):
    def __init__(self, cti_size, wti_size , num_tags_iob , num_tag_ner  , params):
        """
        num_output_rnn : Maximum number of out put for the two iob and ner
        """
        super().__init__()
        self.rnn_iob = rnn(cti_size, wti_size, num_tags_iob , params)
        self.crfiob = crf(num_tags_iob , params)

        self.rnn_ner = rnn(cti_size, wti_size, num_tag_ner , params)
        self.crfner = crf(num_tag_ner , params)
        
        self = self.cuda(ACTIVE_DEVICE) if CUDA else self
        self.HRE = params['HRE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.params = params

    def forward(self, xc, xw, yiob , yner): # for training
        self.zero_grad()
        self.rnn_iob.batch_size = xw.size(0)
        self.rnn_ner.batch_size = xw.size(0)
        self.crfiob.batch_size = xw.size(0)
        self.crfner.batch_size = xw.size(0)

        mask = xw[:,1:].gt(PAD_IDX).float()
        
        h_iob = self.rnn_iob(xc, xw, mask)
        mask_iob = yiob[:, 1:].gt(PAD_IDX).float()
        Ziob = self.crfiob.forward(h_iob, mask_iob)
        
        h_ner = self.rnn_ner(xc, xw, mask)
        mask_ner = yner[:, 1:].gt(PAD_IDX).float()        
        Zner = self.crfner.forward(h_ner, mask_ner)

        scoreiob = self.crfiob.score(h_iob, yiob, mask_iob)
        scorener = self.crfner.score(h_ner, yner, mask_ner)
        #print("score :", score)
        #loss = torch.mean((Ziob + Zner) - (scoreiob + scorener))
        loss = torch.mean(torch.pow((Ziob+Zner) - (scoreiob+scorener) , 2))
        return loss #torch.abs( torch.mean(Ziob - scoreiob) + torch.mean(Zner - scorener)) # NLL loss

    def decode(self, xc, xw, doc_lens): # for inference
        self.rnn_iob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.rnn_ner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfiob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        if self.HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (doc_lens[0] - x) for x in doc_lens])
        else:
            mask = xw.gt(PAD_IDX).float()

        iob_pred = self.rnn_iob(xc, xw, mask)
        #iob_pred = self.iob(h_bio)

        ner_pred = self.rnn_ner(xc, xw, mask)
        #ner_pred = self.ner(h_ner)

        return self.crfiob.decode(iob_pred, mask) , self.crfner.decode(ner_pred, mask)

    def loaddata(self, sentences , cti , wti , itt_iob , itt_ner , yiob=None , yner =None):
        data = dataloader()
        block = []
        for si ,sent in enumerate(sentences) :
            sent = normalize(sent)
            words = tokenize(sent) #sent.split(' ')
            #print(len(words) ,  words)
            x = []
            tokens = []
            for w in words:
                w = normalize(w)
                wxc = [cti[c] if c in cti else UNK_IDX for c in w]
                x.append((wxc , wti[w] if w in wti else UNK_IDX))
                tokens.append(w)
            xc , xw = zip(*x)
            if yiob and yner:
                assert len(yiob[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                assert len(yner[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                block.append((sent,tokens, xc,xw , yiob[si] , yner[si]))    
            else:
                block.append((sent,tokens, xc,xw))
        for s in block:
            if yiob and yner:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=s[4] , yner =s[5])
                data.append_row()
            else:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=[]   , yner =[]  )
                data.append_row()
        data.strip()
        data.sort()
        return data

    def predict(self , sentences , cti , wti , itt_iob , itt_ner):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt_iob : Index To Tag  IOB (Inside Other Begin)
        itt_ner : Index To Tag Named Entity REcognition
        """
        #itt_iob = {v:k for k,v in tti_iob.items()}
        #itt_ner = {v:k for k,v in tti_ner.items()}
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            yiob , yner = self.decode(xc, xw, batch.lens)
            #print(yiob , yner)
            #print([[itt_iob[i] if i in itt_iob else O for i in x] for x in yiob])
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in yiob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in yner])
        data.unsort()
        return data

    def evaluate(self, sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner , parameters = [] ,  model_name = None , save = False , filename = None ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        y0iob : Target values to evaluate (Inside Other Begin)
        y0ner : Target values to evaluate (Named Entity Recognition)
        parameters : 'macro_precision','macro_recall','hmacro_f1', 'amacro_f1','micro_f1','auc'
        """
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1iob , y1ner = self.decode(xc, xw, batch.lens)
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in y1iob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in y1ner])
        data.unsort()
        result_ner = metrics(data.yner , data.y1ner , model_name=model_name + "_ner",save=save,filename=filename)
        result_iob = metrics(data.yiob , data.y1iob , model_name=model_name + "_iob",save=save,filename=filename)
        if parameters:
            print("============ evaluation results ============") 
            for m in parameters:
                if m in result_ner:
                    print("\t" + m +" (ner) = %f"% result_ner[m])
                if m in result_iob:
                    print("\t" + m +" (iob) = %f"% result_iob[m])
        return data
    
# Two task Model NER , Word Segmenting
# this is test based on jupyter RUN_rnn_two_crf
class rnn_two_crf(nn.Module):
    def __init__(self, cti_size, wti_size, num_output_rnn , num_tags_iob , num_tag_ner  , params):
        """
        num_output_rnn : Maximum number of out put for the two iob and ner
        """
        super().__init__()
        #self.Tensor = lambda *x: torch.FloatTensor(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.FloatTensor
        #self.LongTensor = lambda *x: torch.LongTensor(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.LongTensor
        #self.randn = lambda *x: torch.randn(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.randn
        #self.zeros = lambda *x: torch.zeros(*x).cuda(ACTIVE_DEVICE) if CUDA else torch.zeros

        self.rnn = rnn(cti_size, wti_size, num_output_rnn , params)
        self.iob = nn.Linear(num_output_rnn , num_tags_iob)
        self.crfiob = crf(num_tags_iob , params)
        self.ner = nn.Linear(num_output_rnn , num_tag_ner)
        self.crfner = crf(num_tag_ner , params)
        self = self.cuda(ACTIVE_DEVICE) if CUDA else self
        self.HRE = params['HRE']
        self.BATCH_SIZE = params['BATCH_SIZE']
        self.params = params

    def forward(self, xc, xw, yiob , yner): # for training
        self.zero_grad()
        self.rnn.batch_size = xw.size(0)
        self.crfiob.batch_size = xw.size(0)
        self.crfner.batch_size = xw.size(0)

        mask = xw[:,1:].gt(PAD_IDX).float()
        #print("xw", xw.shape)
        h = self.rnn(xc, xw, mask)
        
        mask_iob = yiob[:, 1:].gt(PAD_IDX).float()
        mask_ner = yner[:, 1:].gt(PAD_IDX).float()

        #print("h :" , h.shape)
        iob_fc = self.iob(h)
        Ziob = self.crfiob.forward(iob_fc, mask_iob)
        ner_fc = self.ner(h)
        Zner = self.crfner.forward(ner_fc, mask_ner)
        #print("Y0 :" , y0 , Z)
        #print(Ziob)
        #print(Zner)

        scoreiob = self.crfiob.score(iob_fc, yiob, mask_iob)
        scorener = self.crfner.score(ner_fc, yner, mask_ner)
        

        #print(scoreiob)
        #print(scorener)

        #print("score :", score)
        #yi = torch.mean(scoreiob + scorener)
        #yi_ =  torch.mean(Ziob + Zner)
        loss = torch.mean(torch.pow((Ziob+Zner) - (scoreiob+scorener) , 2))
        #loss = torch.mean((Ziob + Zner) - (scoreiob + scorener))
        #loss = - ((yi* torch.log(yi_)) + ((1-yi) * torch.log(1-yi_)))
        return loss # NLL loss

    def decode(self, xc, xw, doc_lens): # for inference
        self.rnn.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfiob.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        self.crfner.batch_size = len(doc_lens) if self.HRE else xw.size(0)
        if self.HRE:
            mask = Tensor([[1] * x + [PAD_IDX] * (doc_lens[0] - x) for x in doc_lens])
        else:
            mask = xw.gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        
        iob_pred = self.iob(h)

        ner_pred = self.ner(h)

        return self.crfiob.decode(iob_pred, mask) , self.crfner.decode(ner_pred, mask)

    def loaddata(self, sentences , cti , wti , itt_iob , itt_ner , yiob=None , yner =None):
        data = dataloader()
        block = []
        for si ,sent in enumerate(sentences) :
            sent = normalize(sent)
            words = tokenize(sent) #sent.split(' ')
            x = []
            tokens = []
            for w in words:
                w = normalize(w)
                wxc = [cti[c] if c in cti else UNK_IDX for c in w]
                x.append((wxc , wti[w] if w in wti else UNK_IDX))
                tokens.append(w)
            xc , xw = zip(*x)
            if yiob and yner:
                assert len(yiob[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                assert len(yner[si]) == len(xw) , "Tokens length is not the same as Target length (y0)!"
                block.append((sent,tokens, xc,xw , yiob[si] , yner[si]))    
            else:
                block.append((sent,tokens, xc,xw))
        for s in block:
            if yiob and yner:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=s[4] , yner =s[5])
                data.append_row()
            else:
                data.append_item( x0 = [s[0]] , x1 = [s[1]] , xc= [list(s[2])] , xw =[list(s[3])] , yiob=[]   , yner =[]  )
                data.append_row()
        data.strip()
        data.sort()
        return data

    def predict(self , sentences , cti , wti , itt_iob , itt_ner):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt_iob : Index To Tag  IOB (Inside Other Begin)
        itt_ner : Index To Tag Named Entity REcognition
        """
        #itt_iob = {v:k for k,v in tti_iob.items()}
        #itt_ner = {v:k for k,v in tti_ner.items()}
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            yiob , yner = self.decode(xc, xw, batch.lens)
            #print(yiob , yner)
            #print([[itt_iob[i] if i in itt_iob else O for i in x] for x in yiob])
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in yiob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in yner])
        data.unsort()
        return data

    def evaluate(self, sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner , parameters = [] ,  model_name = None , save = False , filename = None ):
        """
        sentences : List of sentence space seperated (tokenization will be done simply by spliting the space between words)
        cti : Character to Index that model trained
        wti : Word to Index that model trained
        itt : Index To Tag (Inside Other Begin)
        y0iob : Target values to evaluate (Inside Other Begin)
        y0ner : Target values to evaluate (Named Entity Recognition)
        parameters : 'macro_precision','macro_recall','hmacro_f1', 'amacro_f1','micro_f1','auc'
        """
        data = self.loaddata(sentences , cti , wti , itt_iob , itt_ner , y0iob , y0ner)
        for batch in data.split(self.BATCH_SIZE, self.HRE):
            xc , xw = data.tensor(batch.xc , batch.xw , batch.lens)
            y1iob , y1ner = self.decode(xc, xw, batch.lens)
            data.y1iob.extend([[itt_iob[i]  for i in x] for x in y1iob])
            data.y1ner.extend([[itt_ner[i]  for i in x] for x in y1ner])
        data.unsort()
        result_ner = metrics(data.yner , data.y1ner , model_name=model_name + "_ner",save=save,filename=filename)
        result_iob = metrics(data.yiob , data.y1iob , model_name=model_name + "_iob",save=save,filename=filename)
        
        if parameters:
            print("============ evaluation results ============") 
            for m in parameters:
                if m in result_ner:
                    print("\t" + m +" (ner) = %f"% result_ner[m])
                if m in result_iob:
                    print("\t" + m +" (iob) = %f"% result_iob[m])
        return data
        

class rnn(nn.Module):
    def __init__(self, cti_size, wti_size, num_tags , params):
        super().__init__()
        self.batch_size = 0
        self.EMBED_SIZE = params['EMBED_SIZE']
        self.HIDDEN_SIZE = params['HIDDEN_SIZE']
        self.NUM_DIRS = params['NUM_DIRS']
        self.NUM_LAYERS = params['NUM_LAYERS']
        self.DROPOUT = params['DROPOUT']
        self.RNN_TYPE = params['RNN_TYPE']
        self.HRE = params['HRE']
        self.params = params

        # architecture
        self.embed = embed(cti_size, wti_size, self.params,  self.HRE)
        self.rnn = getattr(nn, self.RNN_TYPE)(
            input_size = self.EMBED_SIZE,
            hidden_size = self.HIDDEN_SIZE // self.NUM_DIRS,
            num_layers = self.NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = self.DROPOUT,
            bidirectional = (self.NUM_DIRS == 2)
        )
        self.out = nn.Linear(self.HIDDEN_SIZE, num_tags) # RNN output to tag

    def init_state(self, b): # initialize RNN states
        n = self.NUM_LAYERS * self.NUM_DIRS
        h = self.HIDDEN_SIZE // self.NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if self.RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, xc, xw, mask):
        hs = self.init_state(self.batch_size)
        x = self.embed(xc, xw)
        if self.HRE: # [B * doc_len, 1, H] -> [B, doc_len, H]
            x = x.view(self.batch_size, -1, self.EMBED_SIZE)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first = True)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h

class crf(nn.Module):
    def __init__(self, num_tags,params):
        super().__init__()
        self.batch_size = 0
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        #rnd = torch.rand(num_tags, num_tags).cuda(ACTIVE_DEVICE) if CUDA else torch.rand(num_tags, num_tags)
        
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000) # [B, C]
        score[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score # partition function

    def score(self, h, y0, mask): # calculate the score of a given sequence
        score = Tensor(self.batch_size).fill_(0.)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
            trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
            score += (emit_t + trans_t) * mask_t
        last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = LongTensor()
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, SOS_IDX] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b] # best tag
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path
