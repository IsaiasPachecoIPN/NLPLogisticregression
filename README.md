
# NLP Logistic Regression Model for SPAM Classification

This supervised model was developed to classify text messages as SPAM or HAM (legitimate) using logistic regression.

## Methodology:

- **Preprocessing**: The text data was preprocessed by removing stopwords, punctuation, and converting all text to lowercase. Lemmatization was applied to reduce words to their base forms.
- **Vectorization**: A vocabulary was built from the preprocessed text, and the frequency of each term was counted within the dataset.
- **Label Encoding**: SPAM messages were assigned a label of 1, and HAM messages were assigned a label of 0.
- **Model Training**: Logistic regression was performed to determine the optimal weights for each term in the vocabulary, iteratively updating them using a cost function.
- **Model Evaluation**: Performance metrics were calculated to assess the model's accuracy and effectiveness in classifying SPAM and HAM messages.
