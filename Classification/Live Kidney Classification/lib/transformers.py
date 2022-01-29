from lib.utils import transform
import re
import contractions
import emoji
import nltk
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import time
import torch
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss

import matplotlib.pyplot as plt
import seaborn as sns

class cleaner():
    """
    In this class we need to read the data as DataFrames this DataFrame need to have an specific structure which needs have these columns
    OriginBody
    """
    def __init__(self,documents = None):
        if documents is None:
            self.documents = None
        else:
            self.__origintext_columnname = "OriginBody"
            self.__cleans_regexs = [
                    [r'(?<=\w)\s+’\s+(?=\w)' , '\''],
                    [r'(?<=\w)\s+\'\s+(?=\w)' , '\''],
                    # for 
                    [r'\s+:\s+' , ':'],
                    [r'[^A-Za-z0-9]+', ' '],
                ]
            self.documents = documents
            if isinstance(self.documents,pd.DataFrame):
                assert self.__origintext_columnname in self.documents.columns , "make sure the dataframe have column name " + self.__origintext_columnname
            elif isinstance(self.documents, list):
                assert isinstance(self.documents[0], str) , "make sure the document is a list of string"
                self.documents = pd.DataFrame(data = self.documents, columns = [self.__origintext_columnname])

    def set_documents(self, documents):
        self.__cleantext_columnname = "clean_text"
        self.documents = documents
        if isinstance(self.documents,pd.DataFrame):
            assert self.__cleantext_columnname in self.documents.columns , "make sure the dataframe have column name " + self.__cleantext_columnname

    @staticmethod
    def doc_length_distibution(data,column_text):
        print("Process on " + column_text, 'type of' ,data[column_text].dtype )
        document_lengths = []
        if column_text == "clean_text" :
            document_lengths = np.array(list(map(len, data[column_text].str.split())))
        elif column_text == "removed_stopwords" or  column_text == "stem_words":
            document_lengths = np.array(list(map(len, data[column_text])))
        return document_lengths

    def __doc_desc(self,data,column_text):

        print("Process on " + column_text, 'type of' ,data[column_text].dtype )
        document_lengths = []
        if column_text == self.__origintext_columnname or  column_text == "clean_text" :
            document_lengths = np.array(list(map(len, data[column_text].str.split())))
        elif column_text == "removed_stopwords" or  column_text == "stem_words":
            document_lengths = np.array(list(map(len, data[column_text])))

        print("The average number of Words in a document is: {}.".format(np.mean(document_lengths)))
        print("The max number of Words in a document is: {}.".format(np.max(document_lengths)))
        print("The min number of Words in a document is: {}.".format(np.min(document_lengths)))
        fix, ax = plt.subplots(figsize = (15,6))
        ax.set_title('Distribution of number of words on ' + column_text , fontsize = 16)
        ax.set_xlabel("number of words")
        sns.distplot(document_lengths, bins = 50 , ax =ax)
        return document_lengths

    def word_frequency(self, nr_top_words=20):
        tokenized_only_dict = Counter(np.concatenate(self.documents['stem_words'].values))
        tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
        tokenized_only_df.rename(columns={0: 'count'}, inplace = True)
        tokenized_only_df.sort_values('count', ascending=False, inplace=True)
        
        a = tokenized_only_df['count'].values[:nr_top_words]
        amin, amax = min(a) , max(a)
        norm = []

        for i, val in enumerate(a):
            norm.append( (val - amin) / (amax- amin))
        
        return list(zip(tokenized_only_df.index[:nr_top_words] , norm))

    def __word_frequency_barplot(self,df, column_name, nr_top_words=20):
        """ df should have a column named count.
        """
        tokenized_only_dict = Counter(np.concatenate(df[column_name].values))
        tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
        tokenized_only_df.rename(columns={0: 'count'}, inplace = True)
        tokenized_only_df.sort_values('count', ascending=False, inplace=True)
        fig, axs = plt.subplots(1,2,figsize=(20,8))
        
        a = tokenized_only_df['count'].values[:nr_top_words]
        amin, amax = min(a) , max(a)
        norm = []

        for i, val in enumerate(a):
            norm.append( (val - amin) / (amax- amin))

        sns.barplot( norm, list(range(nr_top_words)), palette='hls', orient= 'h', ax=axs[0])
        axs[0].set_yticks(list(range(nr_top_words)))
        axs[0].set_yticklabels(tokenized_only_df.index[:nr_top_words], fontsize=18)
        axs[0].set_title("Word Frequencies " , fontsize=20)
        axs[0].set_xlabel("(a) Frequency of a Word", fontsize = 18)

        document_lengths = []
        if column_name == self.__origintext_columnname or  column_name == "clean_text" :
            document_lengths = np.array(list(map(len, df[column_name].str.split())))
        elif column_name == "removed_stopwords" or  column_name == "stem_words":
            document_lengths = np.array(list(map(len, df[column_name])))

        print("The average number of Words in a document is: {}.".format(np.mean(document_lengths)))
        print("The max number of Words in a document is: {}.".format(np.max(document_lengths)))
        print("The min number of Words in a document is: {}.".format(np.min(document_lengths)))
        axs[1].set_title('Distribution of number of words on ' , fontsize = 20)
        axs[1].set_xlabel("(b) Sentence Length", fontsize = 18)
        sns.distplot(document_lengths, bins = 50 , ax =axs[1])
        plt.show()

    def stats(self):
        print("Stats on " + self.__origintext_columnname)
        self.__doc_desc(self.documents , self.__origintext_columnname)

        if 'clean_text' in self.documents.columns:
            print("Stats on clean_text")
            self.__doc_desc(self.documents , 'clean_text')
        if 'removed_stopwords' in self.documents.columns:
            print("Stats on removed_stopwords")
            #self.__doc_desc(self.documents , 'removed_stopwords')
            self.__word_frequency_barplot(self.documents , 'removed_stopwords')
        if 'stem_words' in self.documents.columns:
            print("Stats on stem_words")
            #self.__doc_desc(self.documents , 'stem_words')
            self.__word_frequency_barplot(self.documents , 'stem_words')

    def clean(self):
        self.documents['clean_text'] = self.documents[self.__origintext_columnname].apply(lambda x : self.__normalize(x))
        self.__demojize(self.documents, 'clean_text','clean_text',False)
        self.__fix_contractions(self.documents, 'clean_text','clean_text') 
        print('Before Removing Duplicate:',self.documents.shape)
        self.__remove_duplicate('clean_text')
        print('After Removing Duplicate:', self.documents.shape)
        for reg, replace in self.__cleans_regexs:
            self.__clean_by_reg(self.documents,reg, replace, 'clean_text','clean_text',False)
        
    def tokenize(self):
        self.__remove_stop_words(self.documents,'clean_text','removed_stopwords')
        self.__stem_words(self.documents, 'removed_stopwords', 'stem_words')

    def __demojize(self,df,source_column,dest_column, verbo = 'True'):
        converted_texts = []
        for i, row in df.iterrows():
            cur_text = row[source_column]
            cur_text_ = emoji.demojize(cur_text)
            if verbo:
                if cur_text_ != cur_text:
                    print(cur_text)
                    print('------------------------------')
                    print(cur_text_)
                    print()
            converted_texts.append(cur_text_)
        df[dest_column] = converted_texts
    
    def __fix_contractions(self,df,source_column,dest_column, verbo = 'True'):
        converted_texts = []
        for i, row in df.iterrows():
            cur_text = row[source_column]
            cur_text_ = []
            for word in cur_text.split():
                cur_text_.append( contractions.fix(word))
            cur_text_ = ' '.join(cur_text_)
            converted_texts.append(cur_text_)
        df[dest_column] = converted_texts
        
    def __normalize(self,x):
        norm = [['&amp;' , 'and'],
                ['&gt;' , ''],
                # Re-tweet
                ['\s+rt' , ' '],
                ['b/c|B/c|B/C|b/C' , 'because'],
                # remove all the html tags
                ['(<script(\s|\S)*?<\/script>)|(<style(\s|\S)*?<\/style>)|(<!--(\s|\S)*?-->)|(<\/?(\s|\S)*?>)', ' '],
                [r'http\S+',' '],
                # remove urls
                [r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?' , ' '],
                [r'@\w+',' '],
                # remove repeated words that more than 3
                [r'(.)\1\1+', '\\1'],
                # remove numbers and the characters that connected to numbers
                # example grandpas portion is about 12k here if we just remove the 12 the number k will remain 
                [r'([A-Za-z]+)?[0-9]+([A-Za-z]+)?', ' '],
                [r'\s+:\s+' , ' ']
            ]
        for reg , repl in norm:
            x = re.sub(reg,repl , x)
        return ' '.join(transform(x.lower()))
        
    def __clean_by_reg(self,df, reg , replace,source_column,dest_column,verbo = True):
        converted_texts = []
        for i, row in df.iterrows():
            cur_text = row[source_column]
            cur_text_ = re.sub(reg , replace,str(cur_text))
            if verbo:
                if cur_text_ != cur_text:
                    print(cur_text)
                    print('------------------------------')
                    print(cur_text_)
                    print()
            converted_texts.append(cur_text_)
        df[dest_column] = converted_texts
        
    def __remove_stop_words(self,df, source_column, dest_column):
        stopwords = nltk.corpus.stopwords.words('english')
        df[dest_column] = list(map(lambda doc: [word for word in doc.split() if word not in stopwords and len(word) >= 2],
                                df[source_column]))
        
    def __stem_words(self,df, source_column, dest_column):
        lemm = nltk.stem.WordNetLemmatizer()
        df[dest_column] = list(map(lambda s : list(map(lemm.lemmatize, s)), df[source_column]))
        stem = nltk.stem.PorterStemmer()
        df[dest_column] = list(map(lambda s : list(map(stem.stem, s)), df[dest_column]))

    def __remove_duplicate(self,source_column):
        self.documents.drop_duplicates(subset=[source_column] , inplace = True)

class frequency():
    def __init__(self,Y, num_class , bias = True):
        self.num_class = num_class
        self.freqs = {}
        self.vocab = {}
        self.bias = bias
        self.vectorizer = self.__vectorizer
        self.targets = Y
        #target to index
        self.tti = { key: i for i,(key,val) in  enumerate(Counter(Y).items())}
        self.itt = { i:key for key,i in self.tti.items() }

    def fit(self,documents):
        assert isinstance(documents, pd.DataFrame) , "The provided document is not DataFrame"
        assert 'stem_words' in documents.columns, "Tokenized words do not existed in the DataFrame make sure you have a column name stem_words in DataFrame"
        
        self.documents = documents
        for i,row in self.documents.iterrows():
            tokens = row['stem_words']
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = np.zeros(self.num_class)
                pair = (t, self.targets[i])
                if pair in self.freqs:
                    self.freqs[pair]+=1
                else:
                    self.freqs[pair] = 1
        for word,_ in self.vocab.items():
            for target,idx in self.tti.items():
                if (word,target) in self.freqs:
                    self.vocab[word][idx] = self.freqs[(word,target)]
        
        data = self.vectorizer()
        return data

    def transform(self, document):
        self.documents = document
        data = self.vectorizer()
        return data

    def __vectorizer(self):
        data = None
        if self.bias:
            data = np.zeros((self.documents.shape[0], 1+self.num_class))
        else:
            data = np.zeros((self.documents.shape[0], self.num_class))
        for i,row in self.documents.iterrows():
            x = None
            if self.bias:
                x = np.zeros((1,1+self.num_class))
                x[0,0] = 1
            else:
                x = np.zeros((1,self.num_class))
            tokens = row['stem_words']
            for word in tokens:
                for target,idx in self.tti.items():
                    if (word,target) in self.freqs:
                        if self.bias:
                            x[0,idx+1] = self.freqs[(word,target)]
                        else:
                            x[0,idx] = self.freqs[(word,target)]
            if self.bias:
                assert(x.shape == (1,1+self.num_class))
            else:
                assert(x.shape == (1,self.num_class))
            data[i,:] = x
        return data

class vectorizer():
    def __init__(self, use_hashing = False, use_tfidf= False, n_features = 2**16 , glove = False , glove_input_file = "", google_w2v = False,
        max_length = 200,
        vec_length = 25):
        self.use_hashing = use_hashing
        self.use_tfidf = use_tfidf
        self.n_feature = n_features
        self.glove = glove 
        self.google_w2v = google_w2v
        self.max_length = max_length
        self.vec_length = vec_length
        if self.use_hashing:
            self.vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,n_features=self.n_feature)
            
        elif self.use_tfidf:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
            # X = self.vectorizer.fit_transform(self.documents['clean_text'])
            # return X
        elif self.glove:
            self.glove_word_2_vec_output = glove_input_file + '.word2vec'
            if(not os.path.exists(self.glove_word_2_vec_output)):
                glove2word2vec(glove_input_file,self.glove_word_2_vec_output)
            self.vectorizer  = KeyedVectors.load_word2vec_format(self.glove_word_2_vec_output , binary=False)
        elif self.google_w2v:
            self.vectorizer = None

    def __sentence2vec(self, documents, alpha = 1e-3):
        vlookup = self.vectorizer.wv.vocab
        vectors = self.vectorizer.wv
        size = self.vectorizer.vector_size
        Z = 0
        for k in vlookup:
            Z+= vlookup[k].count

        output = []
        for i,row in documents.iterrows():
            tokens = row['stem_words']
            v = np.zeros(size, dtype=float)
            count = 0
            for w in tokens:
                if w in vlookup:
                    v+= (alpha / (alpha + (vlookup[w].count / Z))) * vectors[w]
                    count += 1
            if count > 0:
                for i in range(size):
                    v[i] *= 1/count
            output.append(v)
        return np.vstack(output).astype(float)

    def get_weight_matrix(self,embed_size,documents):
        
        vlookup = self.vectorizer.wv.vocab
        vectors = self.vectorizer.wv
        size = self.vectorizer.vector_size
        wti , itw = {} , {}
        assert embed_size <= size, "embedding size is larger than size of vector it should be equal or less than" + str(size)
        for i,row in documents.iterrows():
            tokens = row['stem_words']
            for t in tokens:
                if t not in wti:
                    wti[t] = len(wti)
                    itw[wti[t]] = t
        weight_matrix = np.zeros((len(wti),embed_size))
        for w,i in wti.items():
            try:
                weight_matrix[i] = vectors[w][:embed_size]
            except KeyError:
                weight_matrix[i] = np.random.normal(scale = 0.6, size=(embed_size, ))
        return weight_matrix, wti, itw
        
    def __glove_transform(self,documents):
        uknown_vector = [0.00000000001]*self.vec_length
        X = np.zeros((documents.shape[0], self.max_length*self.vec_length))
        for i,row in documents.iterrows():
            tokens = row['stem_words']
            offset_start = 0
            for t in tokens:
                if t in self.vectorizer:
                    X[i,offset_start:offset_start+self.vec_length] = self.vectorizer[t][:self.vec_length]
                else:
                    X[i,offset_start:offset_start+self.vec_length] = uknown_vector
                offset_start+= self.vec_length    
        return X
    
    def fit(self, documents, column_tokenized):
        assert isinstance(documents, pd.DataFrame) , "The provided document is not DataFrame"
        documents['clean_text_tokenized'] = documents[column_tokenized].apply(lambda x: ' '.join(x))
        if self.use_hashing:
            self.vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,n_features=self.n_feature)
            X = self.vectorizer.transform(documents['clean_text_tokenized'])
        elif self.use_tfidf:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
            X = self.vectorizer.fit_transform(documents['clean_text_tokenized'])
        elif self.glove:
            X = self.__sentence2vec(documents)
        elif self.google_w2v:
            self.vectorizer = Word2Vec(documents[column_tokenized], min_count=1,size=self.vec_length)
            X = self.__sentence2vec(documents)
           
        return X
    
    def transform(self, documents, column_tokenized):
        assert isinstance(documents, pd.DataFrame) , "The provided document is not DataFrame"
        documents['clean_text_tokenized'] = documents[column_tokenized].apply(lambda x: ' '.join(x))
        if self.glove:
            return self.__sentence2vec(documents)
        if self.google_w2v:
            return self.__sentence2vec(documents)            
        else:
            self.vectorizer
            return self.vectorizer.transform(documents['clean_text_tokenized'])

class benchmark():

    def __init__(self,X_train,y_train,X_test,y_test,target_names,feature_names , output_name):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names
        self.feature_names = feature_names
        self.output_name = output_name

    def run(self):
        self.results = []
        try:
            for clf, name in (
                    # (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                    #(Perceptron(max_iter=50), "Perceptron"),
                    # (PassiveAggressiveClassifier(max_iter=50),
                    # "Passive-Aggressive"),
                    (KNeighborsClassifier(n_neighbors=10), "kNN"),
                    (RandomForestClassifier(), "Random forest")):
                #print('=' * 80)
                print(name)
                self.results.append(self.clac(clf))
        except Exception as ex:
            print(ex)

        # try:
        #     for penalty in ["l2", "l1"]:
        #         #print('=' * 80)
        #         print("%s penalty" % penalty.upper())
        #         # Train Liblinear model
        #         self.results.append(self.clac(LinearSVC(penalty=penalty, dual=False,
        #                                     tol=1e-3)))

        #         # Train SGD model
        #         self.results.append(self.clac(SGDClassifier(alpha=.0001, max_iter=50,
        #                                         penalty=penalty)))
        # except Exception as ex:
        #     print(ex)
        
        try:
            # Train SGD with Elastic Net penalty
            # print('=' * 80)
            # print("Elastic-Net penalty")
            # self.results.append(self.clac(SGDClassifier(alpha=.0001, max_iter=50,
            #                                     penalty="elasticnet")))

            # Train NearestCentroid without threshold
            #print('=' * 80)
            print("NearestCentroid (aka Rocchio classifier)")
            self.results.append(self.clac(NearestCentroid()))
        except Exception as ex:
            print(ex)

        try:
            # Train sparse Naive Bayes classifiers
            #print('=' * 80)
            print("Naive Bayes")
            self.results.append(self.clac(MultinomialNB(alpha=.01)))
            #self.results.append(self.clac(BernoulliNB(alpha=.01)))
            #self.results.append(self.clac(ComplementNB(alpha=.1)))
        except Exception as ex:
            print(ex)

        try:
            #print('=' * 80)
            #print("LinearSVC with L1-based feature selection")
            # The smaller C, the stronger the regularization.
            # The more regularization, the more sparsity.
            self.results.append(self.clac(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                            tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))])))
        except Exception as ex:
            print(ex)
        
    def plot(self):
        indices = np.arange(len(self.results))
        res = pd.DataFrame(data=self.results, columns=["clf_names", "F1-Score", "p", "r"])
        res.to_csv(self.output_name)

        self.results = [[x[i] for x in self.results] for i in range(4)]
        clf_names, score, training_time, test_time = self.results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(5, 8))
        plt.title("F1-Score For each Algorithm to Classify Living Donation DataBase ")
        plt.barh(indices, score, .1, label="F1-Score", color='navy')
        #plt.barh(indices + .3, training_time, .2, label="training time",
                #color='c')
        #plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.vlines(np.max(score),0,len(score),color='red')  
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.1)
        plt.subplots_adjust(top=.6)
        plt.subplots_adjust(bottom=.05)
        idx_best = np.argmax(score)
        print('the best model is {0} with average-f1-macro {1}'.format(clf_names[idx_best],score[idx_best]))

        for i, c in zip(indices, clf_names):
            if c == "Pipeline":
                c = "LinearSVC"
            plt.text(-.31, i, c)

        plt.show()

    def clac(self,clf):
        # vectorization
        #print('_'*80)
        #print('training')
        print(clf)
        t0 = time.time()
        clf.fit(self.X_train,self.y_train)
        train_time = time.time() - t0
        #print('test time: %0.3fs'% train_time)
        

        t0 = time.time()
        pred = clf.predict(self.X_test)
        test_time = time.time() - t0
        #print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        score_f1_macro_avg = metrics.f1_score(self.y_test, pred, average='macro')
        score_p = metrics.precision_score(self.y_test, pred)
        score_r = metrics.recall_score(self.y_test, pred)
        print("score_f1_macro_avg:   %0.3f" % score_f1_macro_avg)

        # if hasattr(clf, 'coef_'):
        #     print("dimensionality: %d" % clf.coef_.shape[1])
        #     print("density: %f" % density(clf.coef_))

        #     if self.feature_names is not None:
        #         print("top 10 keywords per class:")
        #         for i, label in enumerate(self.target_names):
        #             top10 = np.argsort(clf.coef_[i])[-10:]
        #             print("%s: %s" % (label, " ".join(self.feature_names[top10])))
        #     print()

        #if opts.print_report:
        #print("classification report:")
        #print(metrics.classification_report(self.y_test, pred))

        #if opts.print_cm:
        #print("confusion matrix:")
        #print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score_f1_macro_avg, score_p, score_r

class helper():
    @staticmethod
    def prediction_report(true_value,pred):
        score = metrics.accuracy_score(true_value, pred)
        score_f1_macro_avg = metrics.f1_score(true_value, pred, average='micro')
        print("accuracy:   %0.3f" % score)

        #if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(true_value, pred))

        #if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(true_value, pred))

        print()

class DataHandler():
    def __init__(self,data , target , vectorizer , test_size = 0.33 , balancer = ""):
        """
        Balancer : smote , 
        """
        self.data = data
        self.target = target
        self.test_size = test_size
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size, random_state = 42)
        print('X_train shape : ', X_train.shape)
        print('Distribution of target in Train data set',Counter(y_train))
        self.train_x , self.train_y = self.prepare(X_train, y_train)
        self.test_x , self.test_y = self.prepare(X_test, y_test)

        self.train_x = vectorizer.fit(X_train)
        self.test_x = vectorizer.transform(X_test)

        X_train_balance = None
        y_train_balance = None
        if balancer  == "":
            number_of_related = min([ v for k,v in Counter(self.train_y).items()])
            for i, x in enumerate(self.train_x):
                if self.train_y[i] == 0 and  number_of_related > 0:
                    X_train_balance.append(x)
                    y_train_balance.append(self.train_y[i])
                    number_of_related-=1
                elif self.train_y[i] == 1:
                    X_train_balance.append(x)
                    y_train_balance.append(self.train_y[i])
        
            
        elif(balancer == "nearmiss"):
            print('=' * 80)
            print('Balanced algorithm is : NearMiss')
            print('_'*80)
            nm = NearMiss(version=2)
            X_train_balance,y_train_balance = nm.fit_resample(self.train_x,self.train_y)

        print('-- Balanced X_train shape : ', X_train_balance.shape)
        print('-- Balanced Distribution of target in Train data set',Counter(y_train_balance))

        print('=' * 80)
        print('X_test shape:', self.test_x.shape)
        print('Distribution of target in Test data set:' , Counter(self.test_y))

        self.train_x = torch.FloatTensor(X_train_balance.toarray())
        self.test_x = torch.FloatTensor(self.test_x.toarray())
        self.train_y = torch.FloatTensor(self.train_y)
        self.test_y = torch.FloatTensor(self.test_y)
        self.len = self.train_x.shape[0]
    @staticmethod
    def balance(X , Y):
        documents_length = cleaner.doc_length_distibution(X , 'stem_words')
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
        print(length_TargetIdxs)
        for l, list_items in length_TargetIdxs.items():
            Length_BalanceTargetIDX_Trains[l] = []
            Length_BalanceTargetIDX_remain[l] = []
            list_items = sorted(list_items,key=lambda x: x[1])
            labels_count = Counter([ label for _,label in list_items ])
            balance_min_length = min(labels_count.values())
            offset = 0
            for label in labels_count.keys():
                Length_BalanceTargetIDX_Trains[label].append(list_items[offset:offset+balance_min_length])
                if(balance_min_length < labels_count[label]):
                    Length_BalanceTargetIDX_remain[label].append(list_items[offset+balance_min_length:])
                offset += labels_count[label]
        X_train_idx_y = []
        for _,records in Length_BalanceTargetIDX_Trains.itmes():
            X_train += records
        return X_train_idx_y
    @staticmethod
    def to_tsv(output_path , data, target , test_size):

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size, random_state = 42, stratify = target)
        print('X_train shape : ', X_train.shape)
        print('Distribution of target in Train data set',Counter(y_train))
        
        
        


        lines_train =[]
        for idx,(i, row) in enumerate(X_train.iterrows()):
            lines_train.append( str(i) + '\t' + str(y_train[idx]) + '\t' + ' ' + '\t' +  row['clean_text'] +  '\n')
        lines_test = []

        for idx,(i, row) in enumerate(X_test.iterrows()):
            lines_test.append( str(i) + '\t' + str(y_test[idx]) + '\t' + ' ' + '\t' +  row['clean_text'] +  '\n')
        fp = open(output_path + "/train.tsv" , 'w')
        fp.writelines(lines_train)
        fp.close()
        fp = open(output_path + "/dev.tsv" , 'w')
        fp.writelines(lines_test)
        fp.close()




    def prepare(self,x,y):
        if y is not None:
                return x, y
        else:
            return x, None

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data , target):
        self.data = data
        self.target = target
        self.len = self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return self.len
    

    



