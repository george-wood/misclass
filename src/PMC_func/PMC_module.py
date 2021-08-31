'''
classes and functions used for texts processing and LDA implementation using gensim.  

'''

import unicodedata
import gensim.corpora as corpora
import pandas as pd

def remove_repeats(doc):
    '''
        Removes Repeats for input document.  Employed in the input_normalization class.
    '''
    s = doc
    i = (s+" "+s).find(s, 1, -1)
    if i == -1:
        doc = doc
    else:
        doc = s[:i-1]
    return(doc)

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
    
    def normalization_remove_repeats(self):
        '''
        input:
            text: raw text as string
        output:
            text_unicode: string
        description (current):
            -removes streing with single identical duplication
        '''           
        self.texts = [remove_repeats(text) for text in self.texts]
        return(self)
        
    def normalization_lower(self):
        '''
        input:
            text: raw text as string
        output:
            text_unicode: string
        description (current):
            -removes capitalizations
        '''        
        self.texts = [text.strip().lower() for text in self.texts]
        return(self)

    def normalization_whitespace(self):
        '''
        input:
            text: raw text as string
        output:
            text_unicode: string
        description (current):
            -removes non space whitespace
        '''
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
        self.bag_of_lem = None
        self.bag_of_text = None
        self.allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        self.length = 2
        self.doc_length = 2
        
    def extract_lemmas(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with bag attribute updated with list of lemmas
        '''
        self.bag_of_lem = [t.lemma_ for t in self.token_list]
        return(self)
    
    def extract_text(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with bag attribute updated with list of lemmas
        '''
        self.bag_of_text = [t.text for t in self.token_list]
        return(self)
    
    def filter_doc_length(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects.  If input object does not meet             document length thrshold, token_list attribute is updated to an empty list.  Otherwise, nothing happens.  Allos the filtering               process to take place between different processing strategies
        '''
        if len(self.token_list) < self.doc_length:
            self.token_list = []
            return(self)
        return(self)
    
    def filter_pos(self):
        '''
        input:
            doc: iterable of spacy token obejcts.  Can be specifically a spacy document object or just a list of tokens
        output:
            self: spacy_filters() object with token_list attribute updated with list of spacy token objects
        '''
        self.token_list = [t for t in self.token_list if t.pos_ in self.allowed_postags]
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
    
    
def gensim_vectorizing(texts,lower=5, upper=.5,restrictedVocabList=[]):
    '''
    input:
        texts: 2-D iterable of list of lemmas for each document
    output:
        id2word: gensim corpora.Dictionary object mapping tokens to ids
        texts: gensim corpora.Dictionary object mapping tokens to ids
        corpus: gensim bow object for entire corpus

    description:
        -processed bag of words listly 2-d vectors into gensim bow-vector objects
    '''
    #build vocabulary, filter vocabulary, calculate metrics
    id2word = corpora.Dictionary(texts)
    initial_vocab_size = len(id2word)
    id2word.filter_extremes(no_below=lower, no_above=upper)
    id2word.filter_tokens(bad_ids=[id2word.token2id[restricted_text] for restricted_text in restrictedVocabList])
    final_vocab_size = len(id2word)
    pct_vocab_kept = final_vocab_size/initial_vocab_size

    corpus = [id2word.doc2bow(text) for text in texts]

    #Print Metrics 
    print('Percentage of Remaining Vocabulary After Filtering Extremes: ', pct_vocab_kept)
    return(corpus, id2word)

def sparcity_calc(corpus, id2word):
    '''
    input:
        corpus: defined corpus gensim object
        id2word: defined id2word gensim object for the corresponding corpus      
    output:
        sparcity: sparcity proportion of the vectorized corpus

    '''
    sparcity = len([item for sublist in corpus for item in sublist])/(len(corpus)*len(id2word))
    return(sparcity)

def corp2dict(row_number,corpus,id2word):
    '''
    input:
        row_number: index of document with in a corpus.  not to be confused with the index from the origional narratives dataframe.
        corpus: defined corpus gensim object
        id2word: defined id2word gensim object for the corresponding corpus      
    output:
        bag_dict: frequency dictionary for the row_number for a particular corpus/id2word
        
    description:
        -returns vectorized document as a dictionary after gensim processing
    '''
    bag_dict = {}
    for id, freq in corpus[row_number]:
        bag_dict[id2word[id]] = freq
    return(bag_dict)


def find_dominiant_topics(lda_model, corpus, texts,raw_texts):
    '''
    input:
        lda_model: Trained LDA model
        corpus: transformed gensim corpus
        texts: 2-D iterable of list of lemmas for each document
        raw_texts: list of raw input strings (for reference)
    description:
        -constructs data frame associating each document with its associated topic and topic contribution
    output:
        -df_dominant_topic: data frame with columns: ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'bag','input_documents']
    '''
    sent_topics_df = pd.DataFrame()

    for i, row in enumerate(lda_model[corpus]):
        sorted_row= sorted(row, key=lambda x: x[1], reverse=True)
        dom_topic = sorted_row[0][0]
        dom_topic_contribution = sorted_row[0][1]
        dom_topic_components = lda_model.show_topic(dom_topic)
        topic_keywords = [word for word, prop in dom_topic_components]
        sent_topics_df = sent_topics_df.append(pd.Series([int(dom_topic), round(dom_topic_contribution,4), topic_keywords]), ignore_index=True)

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    raw_contents = pd.Series(raw_texts)
    sent_topics_df = pd.concat([sent_topics_df, contents,raw_contents], axis=1)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'bag','input_documents']
    return(df_dominant_topic)

def return_top_representatives(dom_topic_df, num_reps = 10):
    '''
    input:
        dom_topic_df: df produced by find_dominiant_topics() function
        num_reps: number of top representatives desired
    description:
        -find top num_reps documents for each topic
    output:
        -top_topic_documents: data frame with top representatives for each topic
    '''
    # Group top 5 sentences under each topic
    top_topic_documents = pd.DataFrame()

    top_topic_documents_grpd = dom_topic_df.groupby('Dominant_Topic')

    for i, grp in top_topic_documents_grpd:
        top_topic_documents = pd.concat([top_topic_documents, 
                                                grp.sort_values(['Topic_Perc_Contrib'], ascending=[0]).head(num_reps)], axis=0)

    # Reset Index    
    top_topic_documents.reset_index(drop=True, inplace=True)
    return(top_topic_documents)