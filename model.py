import  pickle
import  utils
import  numpy               as np
import  matplotlib.pyplot   as plt
import  seaborn             as sns
from    sklearn.metrics     import confusion_matrix

class LogisticRegressor:

    def __init__(self):
        """
        Constructor
        @param data: dataset
        @param vocabulary: vocabulary
        @param word_count_dataset: dataset with the word count
        @param word_count_model: model to predict the score
        """

        self.data                       = None
        self.preproced_data             = None
        self.vocabulary                 = None
        self.word_count                 = None
        self.word_count_probabilities   = None
        self.X                          = None
        self.Y                          = None
        self.weights                    = None

    def load_dataset(self, path, verbose=False):

        """
        Load the dataset from a csv file
        @param path: path to the csv file
        @param verbose: print additional information
        """

        self.data = ""
        for doc in path:
            self.data += open(doc, 'r').read()
        # self.data = [open(doc, 'r', encoding="UTF-8").read() doc for document in path]


    def build_dataset(self, verbose=False):
        """
        Get the X and Y values from the dataset
        """

        #Get the X and Y values
        self.X = []
        self.Y = []

        for doc in self.data.split('\n'):
            doc = doc.split(',')
            # print(f'Doc: {doc}')
            self.X.append(doc[0])
            self.Y.append(1 if doc[1] == 'spam' else 0)

        if verbose:
            print(f'X: {self.X}')
            print(f'Y: {self.Y}')

    def preprocess_text(self, remove_stopwords=True, remove_numbers=True, remove_punctuation=True,lemmatize_text=True,lower_text=True, stop_words_path=None, verbose=False, override=False):

        """
        Function to preprocess the text
        @param remove_stopwords: remove stopwords
        @param remove_numbers: remove numbers
        @param remove_punctuation: remove punctuation
        @param stop_words_path: path to the file with the stopwords
        """

        #Check if the text is already preprocessed
        try:
            if override:
                raise Exception("Override")
            with open('./output/preprocessed_text.txt', 'r') as f:
                self.data = f.read()
            print(f'Preprocessed text loaded')
        except:

            if lower_text:
                print(f'Preprocessing text: lower_text')
                self.X = utils.lower_text(self.X)

            if remove_stopwords:
                print(f'Preprocessing text: remove_stopwords')
                self.X = utils.remove_stopwords(self.X, stop_words_path)

            if remove_punctuation:
                print(f'Preprocessing text: remove_punctuation')
                self.X = utils.remove_punctuation(self.X)

            if remove_numbers:
                print(f'Preprocessing text: remove_numbers')
                self.X = utils.remove_numerical_values(self.X)

            if lemmatize_text:
                print(f'Preprocessing text: lemmatize_text')
                self.X = utils.lemmatize_text(self.X)

            #Save the preprocessed text
            with open('./output/preprocessed_text.txt', 'w') as f:
                for text in self.X:
                    f.write(text.strip() + '\n')

            print(f'Preprocessed text created')

        if verbose:
            print(f'Preprocessed text: {self.X}')



    def build_vocabulary(self, verbose=False, override=False):

        """
        Build the vocabulary from the dataset. If the vocabulary already exists, it will be loaded.
        @param verbose: print additional information
        @param override: override the existing vocabulary
        """

        #Check if vocabulary already exists
        try:
            if override:
                    raise Exception("Override")
            with open('./output/vocabulary.pkl', 'rb') as f:
                self.vocabulary = pickle.load(f)

            print(f'Vocabulary loaded')

        except:
            self.vocabulary = set()
            for sentence in self.X:
                for text in sentence.split():
                    self.vocabulary.add(text)

            #save vocabulary
            with open('./output/vocabulary.pkl', 'wb') as f:
                pickle.dump(self.vocabulary, f)

            print(f'Vocabulary created')

        if verbose:
            print(f'Vocabulary: {self.vocabulary}')

        print(f'Vocabulary size: {len(self.vocabulary)}')

    def word_count_vectorizer(self, verbose=False, override=False):

        """
        Function to create the word count vectorizer
        """

        #Check if the word count already exists
        try:
            if override:
                raise Exception("Override")
            with open('./output/word_count.pkl', 'rb') as f:
                self.word_count = pickle.load(f)
                self.word_count = self.word_count.T
            print(f'Word count loaded')
        except:

            self.word_count = np.zeros((len(self.X), len(self.vocabulary)))

            #Count total words on the X values
            counter = 0 
            m = len(self.Y)

            for i, sentence in enumerate(self.X):
                for j, word in enumerate(self.vocabulary):
                    self.word_count[i, j] = sentence.split().count(word)

            #Save the word count
            with open('./output/word_count.pkl', 'wb') as f:
                pickle.dump(self.word_count, f)

            self.word_count = self.word_count.T
            print(f'Word count created')

        #Prin the sum of the first row
        if verbose:
            print(f'Word count: {self.word_count}')

    def fit(self, num_iterations=1000, learning_rate=0.1, verbose=False, override=False):
        """
        Function to fit the model using X, Y
        """

        #Check if the weights already exists
        try:
            if override:   
                raise Exception("Override")
            with open('./output/weights.pkl', 'rb') as f:
                self.weights = pickle.load(f)
            print(f'Weights loaded')
        except:

            plt.ion()

            plot_x = []
            plot_y = []

            # Crear una figura y un eje
            fig, ax = plt.subplots()

            ax.set_xlim(0, num_iterations)
            ax.set_ylim(0, 10)

            # Línea inicial vacía
            line, = ax.plot(plot_x, plot_y, 'r-')  # 'r-' es el estilo de línea roja

            plt.title('Cost function')
            plt.xlabel('Iterations')
            plt.ylabel('Cost')

            def update_cost(new_x, new_y):
                plot_x.append(new_x)
                plot_y.append(new_y)
                line.set_xdata(plot_x)
                line.set_ydata(plot_y)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

            #initialize the weights with random values
            weights = np.random.rand(len(self.vocabulary))

            #learning rate
            alpha = learning_rate

            self.Y = np.array(self.Y)

            print(f'Weights shape: {weights.shape}')
            print(f'Word count shape: {self.word_count.shape}')
            print(f'Weights: {weights}') 

            m = len(self.Y)

            for i in range(num_iterations):
                
                #Calculate z
                z = np.dot(weights.T, self.word_count)
                
                #Calculate y_hat 
                y_hat = 1 / (1 + np.exp(-z))

                #calculate the cost
                cost = -1* np.sum(self.Y * np.log(y_hat) + (1 - self.Y) * np.log(1 - y_hat))  / m

                #Calculate the gradient
                gradient = np.dot(self.word_count, (y_hat - self.Y).T) / m

                #Update the weights
                weights = weights - alpha * gradient

                update_cost(i, cost)
                print(f'\rCost: {cost}', end='')

            
            self.weights = weights

            print(f'\nWeights calculated: {self.weights}')
            plt.ioff()
            plt.show()

    def predict(self, verbose=False):

        """
        Function to make the preduiction and calculate accuracy
        """
        
        #Calculate z
        z = np.dot(self.weights.T, self.word_count)

        #Calculate y_hat 
        y_hat = 1 / (1 + np.exp(-z))

        y_pred = np.zeros(len(y_hat))
        for i in range(len(y_hat)):
            y_pred[i] = 1 if y_hat[i] > 0.5 else 0

        #Print the prediction
        if verbose:
            for i in range(50):
                print(f'Prediction: {0 if y_hat[i] < 0.5 else 1 } Real: {self.Y[i]}')

        #Calculate the accuracy
        # accuracy = np.sum((y_hat > 0.5) == self.Y) / len(self.Y)

        #calculate precision
        tp = np.sum((y_hat > 0.5) & (self.Y == 1))
        fp = np.sum((y_hat > 0.5) & (self.Y == 0))
        fn = np.sum((y_hat < 0.5) & (self.Y == 1))
        tn = np.sum((y_hat < 0.5) & (self.Y == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

        #Create confusion matrix
        cm = confusion_matrix(self.Y, y_pred)
        sns.heatmap(cm, annot=True, fmt='g',)

        plt.title('Confusion matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Real')
        plt.show()

        if verbose:
            print(f'Accuracy:\t {accuracy}')
            print(f'Precision:\t {precision}')
            print(f'Recall:\t {recall}')
            print(f'F1:\t {f1}')
            