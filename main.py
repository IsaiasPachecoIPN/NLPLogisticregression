from model import *

dataset_path = ['./src/SMS_Spam_Corpus_big.txt']

override = False

nlp = LogisticRegressor()
nlp.load_dataset(dataset_path)
nlp.build_dataset(verbose=True)
nlp.preprocess_text(remove_stopwords=True, remove_numbers=True, remove_punctuation=True,lemmatize_text=True, lower_text=True, stop_words_path='./src/stop_words.txt', verbose=True, override=override)
# nlp.build_vocabulary(override=override, verbose=True)
# nlp.build_background_language_model_probabilities(verbose=True)
# nlp.calculate_em_steps(steps=5, verbose=True)
