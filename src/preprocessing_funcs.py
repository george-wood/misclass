def input_normalization(text):
    '''
    input:
        text: raw text as string
    output:
        text_normalized: normalized string of text
    description (current):
        -lowers all characters
    '''
    text_normalized = text.strip().lower()
    text_normalized = text_normalized.split()
    text_normalized = ' '.join(text_normalized)
    return(text_normalized)


def token_filter(doc):
    '''
    input:
        doc: spaCy doc object
    output:
        list of lemmas: list of lemmas that appear in the doc after filtering
    description:
        -Removes stop words
        -Removes punctuation
        -Removes tokens with length of 1
    '''
    tok_list_working = [t for t in doc if not t.is_stop]
    tok_list_working = [t for t in tok_list_working if (not t.is_punct and not t.is_space)]
    tok_list_working = [t for t in tok_list_working if len(t.text) > 1]
    list_of_lemmas = [t.lemma_ for t in tok_list_working]
    return(list_of_lemmas)


def spacy_tokenizer(text, parser):
    '''
    input:
        text: raw text
        parser: spaCy nlp pipline
    output:
        list_of_lemmas: list of lemmas generated from raw text using the processing pipeline
    description:
        -implements the following helper functions:
            -input_normalization()
            -spaCy pipline (parser)
            -token_filter()
    '''
    text_normalized = input_normalization(text)
    doc = parser(text_normalized)
    list_of_lemmas = token_filter(doc)
    return(list_of_lemmas)