# Police Misconduct Module
Data preprocessing are implemented with a number of classes and function defined in [PMC_module.py](https://github.com/george-wood/misclass/blob/master/src/PMC_func/PMC_module.py).  Full description of all classes and function are availible there.  A listing of the methods are below.

## Preprocessing
- **input normalization()** _(class)_
    - normalization_remove_repeats()
    - normalization_lower()
    - normalization_whitespace()
    - strip_accents()
- **spacy_filters()** _(class)_
    - extract_lemmas()
    - extract_text()
    - filter_doc_length()
    - filter_pos()
    - filter_punc()
    - filter_length()
    - filter_stop()
    
- **gensim_vectorizing()** _(function)_
- **sparcity_calc** _(function)_

## Modeiling and Evaluation
- **corp2dict** _(function)_
- **find_dominiant_topics** _(function)_
- **return_top_representatives** _(function)_