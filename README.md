
# NLP Logistic Regression Model for SPAM Classification

This supervised model was developed to classify text messages as SPAM or HAM (legitimate) using logistic regression.

## Methodology:

- **Preprocessing**: The text data was preprocessed by removing stopwords, punctuation, and converting all text to lowercase. Lemmatization was applied to reduce words to their base forms.
- **Vectorization**: A vocabulary was built from the preprocessed text, and the frequency of each term was counted within the dataset.
- **Label Encoding**: SPAM messages were assigned a label of 1, and HAM messages were assigned a label of 0.
- **Model Training**: Logistic regression was performed to determine the optimal weights for each term in the vocabulary, iteratively updating them using a cost function.
- **Model Evaluation**: Performance metrics were calculated to assess the model's accuracy and effectiveness in classifying SPAM and HAM messages.


## Installation

In order tu run the program, some python packages have to be installed

```bash
  pip install numpy matplotlib seaborn scikit-learn
```

Then run 

```bash
  python main.py
```


## Usage/Examples

```python
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

```


## Output

![Alt text](https://github.com/IsaiasPachecoIPN/NLPLogisticregression/blob/main/images/Figure_1.png "Cost Function")

![Alt text](https://github.com/IsaiasPachecoIPN/NLPLogisticregression/blob/main/images/Figure_2.png "Confusion Matrix")

![Alt text](https://github.com/IsaiasPachecoIPN/NLPLogisticregression/blob/main/images/output.jpg "Output Example")

![Alt text](https://github.com/IsaiasPachecoIPN/NLPLogisticregression/blob/main/images/output_2.jpg "Output Example")
