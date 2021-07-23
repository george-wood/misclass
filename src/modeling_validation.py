def gensim_vectorizing(texts,lower=5, upper=.5):
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
    final_vocab_size = len(id2word)
    pct_vocab_kept = final_vocab_size/initial_vocab_size

    corpus = [id2word.doc2bow(text) for text in lemmatized_texts]

    #Print Metrics 
    print('Percentage of Remaining Vocabulary After Filtering Extremes: ', pct_vocab_kept)
    return(corpus, id2word)

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

def return_top_representatives(dom_topic_df = df_dominant_topic, num_reps = 10):
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

    top_topic_documents_grpd = df_dominant_topic.groupby('Dominant_Topic')

    for i, grp in top_topic_documents_grpd:
        top_topic_documents = pd.concat([top_topic_documents, 
                                                grp.sort_values(['Topic_Perc_Contrib'], ascending=[0]).head(num_reps)], axis=0)

    # Reset Index    
    top_topic_documents.reset_index(drop=True, inplace=True)
    return(top_topic_documents)