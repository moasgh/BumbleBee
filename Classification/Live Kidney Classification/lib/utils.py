import re
from time import time
from glob import glob
from os.path import isfile
import pandas as pd
import numpy as np

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