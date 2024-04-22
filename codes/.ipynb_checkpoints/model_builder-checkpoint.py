import pandas as analytics
import numpy as maths
import tensorflow as tf
import matplotlib.pyplot as graph
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU , Dense
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

class model_builder :

    def __init__(self,architecture, data):

        self.architecture = architecture
        self.data = data
    

        if self.architecture["model"] == None :
            self.training = False
            
            print("\nExtracting training corpus .... ",end = " ")
            training_corpus = self.extract_corpus(self.data["train"])
            
            if not len(training_corpus) : 
                print("Training corpus couldnt be extracted !")
            
            else : 
                print("Training corpus extracted successfully !")
                print("\n\nTokenising the corpus .... ")
                self.tokenise(training_corpus)
            
                try :
                
                    if self.tokenizer : 
                        print("Corpus be tokenised successfully !")
                        print("\n\nGenerating input sequence .... ", end = " ")
                        self.generate_input_sequence(training_corpus)
                    
                         
                        if self.input_sequences : 
                            print("Input sequence generated successfully !")
                            # self.plot_histogram()
                            self.compile_model()

                            if self.architecture_defined :
                                print("Modelled compiled successfully with "+self.architecture["type_of_cells"].upper()+" cells!")
                                print("Fit for training ! Training can be done !!!")

                            else :
                                print("Model cant be compiled ! Not fit for training !!!")
                            

                                
                except NameError as error:
                    print("\n\n",error)

                except TypeError as error:
                    print("\n\n",error)

                except ValueError as error:
                    print("\n\n",error)
                

        
        

        
    def extract_corpus(self,df):
        """The corpus is extracted from the dataframe provided. The dataframe must have a column with column name 'data' from which the data would be extracted"""
        corpus = []
        for i in range(len(df)):
            text = df.iloc[i]['data']
            corpus.append(text)
        return corpus
    
    def tokenise(self,corpus):
        """The corpus passed is tokenised. Corpus must be in the form of a list."""
        # Tokenize the corpus
        if not self.training :
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
            self.tokenizer.fit_on_texts(corpus)
            self.total_words = len(self.tokenizer.word_index) + 1
            print("Total Number of Unique Words [VOCABULARY] :",self.total_words)
        else :
            print("Training is already done ! No need to tokenize corpus again !")
    
    def generate_input_sequence(self, corpus):
        """Input sequence based on the corpus is generated"""
        # Generate input sequences
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        if not self.training :
            self.input_sequences = input_sequences
        else :
            return input_sequences

    
    def find_optimal_sequence_len(self):
        """The optimal sequence len from the input sequence which would make the embedding input is found. It returns the length value at which almost 99% of data is satisfied"""
        length_vectors = maths.array([len(x) for x in self.input_sequences])
        unique ,counts = maths.unique(length_vectors, return_counts=True)
        cumm_sum = counts.cumsum()
        total = cumm_sum[-1]
        cumm_sum = cumm_sum/total
        threshold_value = list(filter(lambda x: x if x > 0.99 else 0, cumm_sum))[0]
        index_value = list(cumm_sum).index(threshold_value)
        optimal_sequence_len = unique[index_value]
        self.optimal_sequence_len = optimal_sequence_len
        print("Optimal Sequence Length :",self.optimal_sequence_len)
        
        graph.scatter(range(len(length_vectors)),length_vectors, label = "Vector Lengths", alpha = 0.3, marker = ".")
        max_vector = len(length_vectors) * [optimal_sequence_len]
        graph.plot(max_vector, c = 'red', label = 'Selected Value')
        graph.ylabel('Length of vector')
        graph.xlabel('Input Sequences')
        graph.title("Length of vectors vs Input Sequences")
        graph.legend()
        graph.plot()
    
        # Pad sequences
        self.input_sequences = maths.array(tf.keras.preprocessing.sequence.pad_sequences(self.input_sequences, maxlen=self.optimal_sequence_len, padding='pre'))
    
    def plot_histogram(self):
        """Plot histogram if need to verify the optimal sequence length. If called , then that must be before training"""
        if len(self.input_sequences[0]) != len(self.input_sequences[1]):
            length_vectors = maths.array([len(x) for x in self.input_sequences])
            graph.hist(length_vectors)
            graph.xlabel('Length of vector')
            graph.ylabel('Input Sequences')
            graph.title("Histogram of length of vectors vs Input Sequences")
            # graph.plot()
        else :
            print("Input Sequence is padded !")


    def compile_model(self):
        """Model is compiled based on the architecture defined by the user"""
        self.find_optimal_sequence_len()

        print("\n\nCompiling model .... ", end = " ")
        
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, self.architecture['embedding_dimension'], input_length=self.optimal_sequence_len-1))
        
        architecture_defined = False
        if self.architecture['type_of_cells'] == "lstm" :
            self.model.add(LSTM(self.architecture['number_of_cells'], dropout = self.architecture['dropout_percentage']))
            architecture_defined = True
            
        elif self.architecture['type_of_cells'] == "gru" :
            self.model.add(GRU(self.architecture['number_of_cells'], dropout = self.architecture['dropout_percentage']))
            architecture_defined = True

        else : 
            print("\n Choose either GRU or LSTM as the type of cells" )

        if architecture_defined :
            self.model.add(Dense(self.total_words, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.architecture_defined = architecture_defined


    def train_model(self):
        
        """The data is broken as per convenience of the user and model is trained"""
        if self.architecture_defined :
            # data division
            X, y = self.input_sequences[:,:-1],self.input_sequences[:,-1]
            y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)

            self.X = X
            self.y = y
            
    
            # Model compilation and training    
            start_time = time.time()
            self.model.fit(X, y, epochs = self.architecture['max_epochs'], verbose=1)
            end_time = time.time()
            
            time_taken = end_time - start_time 
            time_taken = time_taken/3600
            print("Total Training time : %.2f hrs"%time_taken)
            print("Modelled trained successfully !")
            self.architecture["model"] = self.model
            self.training = True

        else :
            print("Not fit for training !")
        

    def test_model(self):

        if self.architecture["model"] :

            if len(self.data["test"]) > 0 :
                test_corpus = self.extract_corpus(self.data["test"])
                test_input_sequences = self.generate_input_sequence(test_corpus)
                
                # Pad sequences
                test_input_sequences = maths.array(tf.keras.preprocessing.sequence.pad_sequences(test_input_sequences, maxlen=self.optimal_sequence_len, padding='pre'))
                
                # Create predictors and labels
                test_X, test_y = test_input_sequences[:,:-1],test_input_sequences[:,-1]
                test_y = tf.keras.utils.to_categorical(test_y, num_classes=self.total_words)
                loss , accuracy = self.model.evaluate(test_X, test_y)
                print("Loss in testing : %.3f"%loss)
                print("Accuracy in testing :%.3f"%accuracy)

            else:
                print("Test data is not available passed !")

        
        else :
            print("Model is not defined !")

class model_tester :
    def __init__(self, training_path, model_path):

        self.corpus = self.extract_corpus(training_path)
        self.model = self.get_model(model_path)
        # self.interact()
        
    def extract_corpus(self,training_path):
        df_train = analytics.read_csv(training_path)
        corpus = []
        for i in range(len(df_train)):
            text = df_train.iloc[i]['data']
            corpus.append(text)
        return corpus

    def get_model(self,model_path):
        model = load_model(model_path)  
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(self.corpus)  
        self.vocab_size = len(self.tokenizer.word_index) + 1
        return model

    def generate_response(self,input_text, history=[]):
        input_seq = self.tokenizer.texts_to_sequences([input_text])[0]
        max_sequence_len = 100
        input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = self.model.predict(input_seq)
        indices = []
        while len(indices) < 2*len(input_text):
            indices.append(maths.argmax(predicted_probs))
            predicted_probs[0][indices[-1]] = 0
        predicted_word = ""
        for predicted_word_index in indices:
               predicted_word = predicted_word + " " + self.tokenizer.index_word.get(predicted_word_index, "")    
        return predicted_word, history


    def interact(self):
        history = []
        print("Chatbot: Hello! How can I assist you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break
            response, history = self.generate_response(user_input, history)
            print("Chatbot:", response)
        