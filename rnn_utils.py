import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
np.random.seed(4)
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from utils import *
from sklearn.preprocessing import LabelEncoder

def read_csv(filename = './data/reuters_result_scored.csv'):
    name = []
    text =[]
    date = []
    score = []

    with open (filename,encoding="utf8") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)

        for row in csvReader:
            name.append(row[0])
            text.append(row[1])
            date.append(row[2])
            score.append(row[3])
    X = np.column_stack([name, text, date])
    Y = np.asarray(score, dtype=int)

    return X, Y


def convert_to_one_hot(Y, C):
    
    #targets = np.array(Y).reshape(-1)
    #Y = np.eye(C)[targets]
    le = LabelEncoder()
    Y = le.fit_transform(Y)    
    return Y

def read_hotel_embeddings(model = './hotel_embedding.h5'):
    hotel_embedding_model = load_model(model)
    hotel_ids = load_obj('hotel_index').keys()
    hotel_to_vec_map = extract_weights('hotel_embedding', hotel_embedding_model)
    
    hotels_to_index = load_obj('hotel_index')
    index_to_hotels = load_obj('index_hotel')
        
    return hotels_to_index, index_to_hotels, hotel_to_vec_map


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sentence_to_avg(sentence, word_to_vec_map, hotels_to_index):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the word2vec representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 300-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (300,)
    """
    
    #Split sentence into list of lower case words
    #sentence = sentence[2]
    #sentence = sentence[0]

    # Initialize the average word vector.
    avg = np.zeros((len(sentence),1))
    
    # average the word vectors. 
    total = 0
    exist = 0
    for h in sentence:
        try:
            total += hotel_to_vec_map[hotels_to_index[h]]
            exist += 1
        except:
            pass

    avg = total/exist 
    return avg


def predict(X, Y, W, b, n=1000):
    """
    Given X (sentences) and Y (sentiment indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label sentiment, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    submissions = []
    hotels_to_index, index_to_hotels, hotel_to_vec_map = read_hotel_embeddings()

    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        #words = X[j][2].lower().split()
        hotels = X[j]
        # Average words' vectors
        avg = np.zeros((hotel_to_vec_map[5101].shape[0] ,))    
        exist = 0
        for h in hotels:
            try:
                avg += hotel_to_vec_map[hotels_to_index[h]]
                exist += 1

            except:
                pass
        avg = avg/exist 

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        submission_indices = np.argpartition(A, -n)[-n:]
        submission = submission_indices[np.argsort(A[submission_indices])]
        submissions.append(submission)

    #data = np.concatenate((pred, Y.reshape(Y.shape[0],1)), axis=1)
    #df = pd.DataFrame(data = data, columns=['pred', 'actual'])
    #df.to_csv("./data/to_check.csv", index= False)
    print("MRR: "  + str(mrr(Y, submissions,index_to_hotels)))
    
    return submissions


def model(X, Y, hotel_to_vec_map, learning_rate = 0.005, num_iterations = 1000):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    hotels_to_index = read_hotel_embeddings()[0]
    # Define number of training examples
    m = X.shape[0]                          # number of training examples
    n_y = len(hotels_to_index)              # number of classes  
    n_h = hotel_to_vec_map[5101].shape[0]                                 # dimensions of the GloVe vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    batch_size = 32
    steps = m//batch_size
    # Optimization loop
    for t in range(num_iterations): # Loop over the number of iterations
        submissions = []

        for step in range(steps-1):
            # Convert Y to Y_onehot with n_y classes
            try: 
                Y_batch = Y[step*batch_size:(step+1)*batch_size]
                n_y = np.unique(Y_batch)
            except:
                Y_batch = Y[step*batch_size:]
                n_y = np.unique(Y_batch)
            
            Y_oh = convert_to_one_hot(Y_batch, C = n_y) 
            m = Y_batch.shape[0]
            
            for i in range(m):          # Loop over the training examples
                
                # Average the word vectors of the words from the i'th training example
                avg = sentence_to_avg(X[step*batch_size+i][:], hotel_to_vec_map, hotels_to_index)
    
                # Forward propagate the avg through the softmax layer
                z = np.dot(W, avg) + b
                a = softmax(z)
    
                # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
                cost = -np.sum(Y_oh[i] * np.log(a))
                
                # Compute gradients 
                dz = a - Y_oh[i]
                dW = np.dot(dz.reshape((-1,1)), avg.reshape((1, -1)))
                db = dz
    
                # Update parameters with Stochastic Gradient Descent
                W = W - learning_rate * dW
                b = b - learning_rate * db
        print(t)
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            submission = predict(X, Y, W, b) 
            
        submissions.append(submission)  
            
    return submissions, W, b

def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def sentences_to_indices(X, word_to_index, max_len=24):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros.
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training news in lower case and split is into words.
        #sentence_words = X[i][2].lower().split()
        sentence_hotels = X[i][0] #### not sure X[i] de olabilir
        j = 0
        
        for h in sentence_hotels:
            X_indices[i, j] = hotel_to_index[h]
            j +=1
                
    return X_indices


def pretrained_embedding_layer(hotel_to_vec_map, hotel_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                 
    emb_dim = hotel_to_vec_map[5101].shape[0]      # define dimensionality of word2vec vectors (= 300)
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
   
    for hotel, idx in hotel_to_index.items():
        emb_matrix[idx, :] = hotel_to_vec_map[hotel]

    embedding_layer = Embedding(input_dim = vocab_len, output_dim = emb_dim, trainable = False)
 
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def with_LSTM(input_shape, hotel_to_vec_map, hotel_to_index):
    """
    Function creating the Sentiment Analysis with Long Short Term Memory model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 300-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (~ 900 million)

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with word2vec Vectors 
    embedding_layer = pretrained_embedding_layer(hotel_to_vec_map, hotel_to_index)
    
    embeddings = embedding_layer(sentence_indices)   
    
    # Propagate the embeddings through an LSTM layer with input_len+1-dimensional hidden state
    X = LSTM(24, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(24, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(len(hotel_to_index.keys()))(X)
    X = Activation('softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model