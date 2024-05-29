import spacy
import es_core_news_sm

#Preprocessing functions
def remove_stopwords(text_array, stop_words_path):

    """
    Function to remove the stopwords from a text
    @param text:    The text to remove the stopwords
    @param stop_words_path: path to the file with the stopwords
    @return:        The text without the stopwords
    """

    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = f.read()
        stop_words = stop_words.split(',')
        stop_words = [word.replace('"', '').strip() for word in stop_words]

    # print(f'Stop words: {stop_words}')

    # return ' '.join([word for word in text.split() if word not in stop_words])
    X = []
    for text in text_array:
        X.append(' '.join([word for word in text.split() if word not in stop_words]))

    return X

def lower_text(text_array):
    """
    Function to lower the text
    @param text:    The text to lower
    @return:        The text lowered
    """
    X = []
    for text in text_array:
        X.append(text.lower())

    return X

def remove_punctuation(text_array):
    """
    Function to remove the punctuation from a text
    @param text:    The text to remove the punctuation
    @return:        The text without the punctuation
    """
    
    puntuation_signs = ['.',',',';',':','!','¡','¿','?','(',')','[',']','{','}','"','/','\\','|','<','>','@','#','$','%','^','&','*','_','+','-','=','~','`']

    X = []
    for text in text_array:
        X.append(''.join([i for i in text if i not in puntuation_signs]))
    
    return X

def remove_numerical_values(text_array):
    """
    Function to remove the numerical values from a text
    @param text:    The text to remove the numerical values
    @return:        The text without the numerical values
    """
    X = []
    for text in text_array:
        X.append(''.join([i for i in text if not i.isdigit()]))
    
    return X

def lemmatize_text(text_array, verbose=False):
    """
    Function to lemmatize the text
    @param text:    The text to lemmatize
    @return:        The lemmatized text
    """

    nlp = spacy.load('en_core_web_sm')
    nlp = es_core_news_sm.load()

    # nlp = spacy.load('en', disable = ['parser','ner'])
    # nlp = spacy.load("en_core_web_sm")

    X = []
    for text in text_array:
        doc = nlp(text)
        X.append(' '.join([token.lemma_ for token in doc]))

    if verbose:
        print(f'Lemmatized text: {X}')

    return X