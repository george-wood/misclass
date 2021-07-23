class input_normalization():
    '''
        Class for storing input normalization techniques:
        Current list of methods.  More to come.
        All functions take a list of strings as inputs (documents).
        All functions out put a list of strings as outputs (documents).
        Allows for the easy experimentation of different pre-processing techniques.
    '''
    
    def __init__(self, texts = None):
        self.texts = texts
        
    def normalization_lower(self):
        self.texts = [text.strip().lower() for text in self.texts]
        return(self)

    def normalization_whitespace(self):
        self.texts = [text.split() for text in self.texts]
        self.texts = [' '.join(text) for text in self.texts]
        return(self)

    def strip_accents(self):
        '''
        input:
            text: raw text as string
        output:
            text_unicode: string converted to unicode
        description (current):
            -removes accents and other non-ascii stuff
        '''
        strip_accents_list = []
        for text in self.texts:
            try:
                text = unicode(text, 'utf-8')
            except NameError: # unicode is a default on python 3 
                pass

            text = unicodedata.normalize('NFD', text)\
                .encode('ascii', 'ignore')\
                .decode("utf-8")

            strip_accents_list.append(str(text))

        self.texts = strip_accents_list
        return(self)


class spacy_filters():
    '''
        Class for storing methods for filtering spacy documents.
        Running list of methods.
        Attributtes:
            -doc: Spacy input document, should be the the origional spacy document object from the stram and remain un altered
            -token_list: list of spacy token objects.  updated by each functions
            -bag: list of words storing final lemmatized text
            -allowed_postags: list of POS tags to be kept in the filter_pos() method
            -length: token length threshold for  filter_length() method
    '''
    
    def __init__(self, doc = None):
        self.doc = doc
        self.token_list = doc
        self.bag = None
        self.allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        self.length = 1
        
    def lemmatization(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with bag attribute updated with list of lemmas
        '''
        self.bag = [t.lemma_ for t in self.token_list]
        return(self)

    def filter_pos(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects
        '''
        self.token_list = [t.lemma_ for t in self.token_list if t.pos_ in self.allowed_postags]
        return(self)

    def filter_punc(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects
        '''
        self.token_list = [t for t in self.token_list if (not t.is_punct and not t.is_space)]
        return(self)

    def filter_length(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects
        '''
        self.token_list = [t for t in self.token_list if len(t.text) > self.length]
        return(self)

    def filter_stop(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects
        '''
        self.token_list = [t for t in self.token_list if not t.is_stop]
        return(self)


'''
Example Processing:
'''

#python basics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import unicodedata


#spacy stuff
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
import en_core_web_sm

#other
import pickle
import re

#Import Data
path = "update local path"
narratives = pd.read_csv(path + "/misclass/data/narratives.csv")
intake = narratives.column_name.str.contains('take')
narratives = (narratives[intake])[["cr_id", "column_name", "text"]]
narratives = narratives.drop_duplicates()
df = narratives[:].copy()
df_list = df.text.values.tolist() #store documents as list of lists


#stream spacy docs into lemmatization functions
nlp = spacy.load('en_core_web_sm')

lemmatized_texts = []
for doc in nlp.pipe(df_list_normalized, batch_size=20):
    spacy_tokenizer_test = spacy_filters(doc = doc)
    lemmatized_doc = spacy_tokenizer_test.filter_length()\
                                .filter_stop()\
                                    .filter_punc()\
                                        .lemmatization()\
                                            .bag

    lemmatized_texts.append(lemmatized_doc)
