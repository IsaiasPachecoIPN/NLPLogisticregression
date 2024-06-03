from model import *

dataset_path = ['./src/SMS_Spam_Corpus_big.txt']

override = False

nlp = LogisticRegressor()
nlp.load_dataset(dataset_path)
nlp.build_dataset(verbose=False)
nlp.preprocess_text(remove_stopwords=True, remove_numbers=False, remove_punctuation=True,lemmatize_text=True, lower_text=True, stop_words_path='./src/stop_words.txt', verbose=False, override=False)
nlp.build_vocabulary(override=override, verbose=False)
nlp.word_count_vectorizer(verbose=False, override=False)
nlp.fit(num_iterations=1000, learning_rate=0.40, verbose=True, override=True)
nlp.predict(verbose=True)
# nlp.build_background_language_model_probabilities(verbose=True)
# nlp.calculate_em_steps(steps=5, verbose=True)
