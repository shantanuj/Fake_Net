'''To do:
1) Encode spatial and sequence using ConvLSTM
2) Encode attention of previous sequences
3) Also see http://athena.ecs.csus.edu/~millerk/ 
https://web.stanford.edu/class/cs224n/reports/2760496.pdf

'''

train_file_path = "data_stance/combined_train_stance.csv"
test_file_path = "data_stance/combined_test_stance.csv"

from __future__ import print_function
from keras.models import model_from_json
import pandas as pd
import csv
import json
from scipy.sparse import hstack
import os
import numpy as np
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Embedding
from gensim.models import KeyedVectors
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, concatenate
from keras.models import Model
from sklearn import metrics
from sklearn.model_selection import train_test_split

'''Text processing functions'''
def tokenize_text(q, lower= True):
    #Obtain word tokens 
    try:
        tokens = word_tokenize(q.decode('utf-8'))
    #except UnicodeDecodeError:
     #   tokens = word_tokenize(q.decode('utf-8'))
    except:
        tokens = ["<UNK>"]
    #print(q)
    word_tokens = [word for word in tokens if word.isalpha()]  #only include words; not sure if best option
    word_tokens = [word for word in word_tokens if word not in stop_words]
    if(lower):
        word_tokens = map(lambda x: x.lower(), word_tokens) #converting all to lower case
    return word_tokens


seq_length_list = []

def get_word_to_int_sequence(tokens):
    '''Returns sequence and updates vocab'''
    '''Does increasing number of functions impact performance?'''
    seq = []
    #global max_seq_length
    for token in tokens:
        if(token not in word_to_int):
            word_to_int[token] = len(word_to_int)
            int_to_word[len(word_to_int)] = token
            seq.append(word_to_int[token])
        else:
            seq.append(word_to_int[token])
    #if(len(seq)>max_seq_length):
     #   max_seq_length = len(seq)
    seq_length_list.append(len(seq))
    return seq


'''Loading functions'''
wordvec_dir = "../word_vectors/corpus_relevant_vectors.txt"
vocab_dir = "vocab.json"
word_embedding_dim = 300

def load_word_vocab(path="vocab.json"):
    with open(path, 'r') as fp:
        word_to_int = json.load(fp)
        int_to_word = {i:word for word,i in word_to_int.items()}
    return word_to_int, int_to_word

word_to_int, int_to_word = load_word_vocab(vocab_dir)
word_to_int['<UNK>'] = len(word_to_int)
int_to_word[len(int_to_word)] = "<UNK>"
word_tokens_in_corpus = word_to_int.keys()

print("Loaded vocab with {} tokens ".format(len(word_tokens_in_corpus)))


print("Loading word embeddings")
'''Load word vectors in keras format'''

#Remember word_to_int gives all words to integers for all tokens
num_words_in_corpus = len(word_to_int)
vocab = word_to_int.keys()
#word_to_embedding = {}

wordint_to_embedding = np.random.randn(len(word_to_int), word_embedding_dim) #for unknown words we assume random values
wordint_to_embedding[-1] = 0 #unknown token is 0 
word_vectors = KeyedVectors.load_word2vec_format(wordvec_dir, binary=False)
for word in word_tokens_in_corpus:
    if(word in word_vectors.vocab):
        wordint_to_embedding[word_to_int[word]] = word_vectors.word_vec(word)
    #try:
     #   embedding = word_vectors[word]
      #  wordint_to_embedding[word_to_int[word]] = embedding
    #except:
     #   continue
print("Intialized word embeddings")


label_map = {'agree':0,'disagree':1, 'discuss':2, 'unrelated':3}
'''Stance training set'''
#Need to segment into headline, body, label
training_df = pd.read_csv(train_file_path)
train_headlines = training_df['Headline'].tolist()
train_articles = training_df['Body'].tolist() #store each sequence in list
train_labels = map(lambda x: label_map[x],training_df['Stance'].tolist())
#convert labels to one hot encoded 
train_labels = keras.utils.to_categorical(np.asarray(train_labels))


'''Stance testing set'''
test_df = pd.read_csv(test_file_path)
test_headlines = test_df['Headline'].tolist()
test_articles = test_df['Body'].tolist() #store each sequence in list
test_labels = map(lambda x: label_map[x],test_df['Stance'].tolist())
test_labels = keras.utils.to_categorical(np.asarray(test_labels))


'''TFIDF'''
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_headlines = TfidfVectorizer(stop_words=stop_words,ngram_range=(1,2),max_df= 0.90, min_df= 0.01, decode_error ="replace")
#headlines_tfidf = tfidf.fit_transform(train_headlines+test_headlines)
train_headlines_tfidf = tfidf_headlines.fit_transform(train_headlines).todense()
test_headlines_tfidf = tfidf_headlines.transform(test_headlines).todense()

tfidf_articles = TfidfVectorizer(stop_words=stop_words,ngram_range=(1,2),max_df= 0.90, min_df= 0.01, decode_error ="replace")
train_articles_tfidf = tfidf_articles.fit_transform(train_articles).todense()
test_articles_tfidf = tfidf_articles.transform(test_articles).todense()

concatenated_train_tfidf= np.concatenate((train_headlines_tfidf, train_articles_tfidf),axis=-1)
concatenated_test_tfidf= np.concatenate((test_headlines_tfidf, test_articles_tfidf),axis=-1)

dims_tfidf = concatenated_test_tfidf.shape[-1] 

'''Storing TFIDF as a pickle'''
with open('headline_tfidf_vectorizer.pk', 'wb') as fin:
    pickle.dump(tfidf_headlines, fin)
with open('articles_tfidf_vectorizer.pk', 'wb') as fin:
    pickle.dump(tfidf_articles, fin)
with open('tf_idf_dims.txt','w') as f:
    f.write(str(dims_tfidf))

print("Stored TFIDF for headlines and articles as pickle files")
    
#Parameters 
max_head_length = 30 #assumption max headline will be 30 words
max_article_length = 200 #assumption
num_hidden_units_LSTM = 64

'''Convert training and testing headlines and articles to sequences for RNN model'''
print("Mapping training and testing headlines and articles to integer sequences")
train_headline_sequences = map(lambda x: get_word_to_int_sequence(tokenize_text(x)),train_headlines) #converts each sentence to sequence of words
train_article_sequences =  map(lambda x: get_word_to_int_sequence(tokenize_text(x)),train_articles)
test_headline_sequences =  map(lambda x: get_word_to_int_sequence(tokenize_text(x)),test_headlines)
test_articles_sequences =  map(lambda x: get_word_to_int_sequence(tokenize_text(x)),test_articles)

X_train_headline_sequences = pad_sequences(train_headline_sequences, maxlen= max_head_length)#perform padding for a sequence of max length
X_train_article_sequences = pad_sequences(train_article_sequences, maxlen = max_article_length)
X_test_headline_sequences = pad_sequences(test_headline_sequences, maxlen = max_head_length)
X_test_article_sequences = pad_sequences(test_articles_sequences, maxlen = max_article_length )
'''
Core model:
1) Create embedding layers (lookup)
These are trainable and hence both article and header will have different trained embeddings (trainable is true)
2) Bidirectional LSTM with return sequences (outputs across all seq)
3) Dropout tuning
'''
headline_embedding_layer = Embedding(num_words_in_corpus, word_embedding_dim, weights = [wordint_to_embedding], input_length = max_head_length, trainable = True)
article_embedding_layer = Embedding(num_words_in_corpus, word_embedding_dim, weights = [wordint_to_embedding], input_length = max_article_length, trainable = True)

headline_seq_placeholder = Input(shape=(max_head_length,), dtype = 'int32')
headline_embedded_sequence = headline_embedding_layer(headline_seq_placeholder)
article_seq_placeholder = Input(shape=(max_article_length,), dtype = 'int32')
tfidf_placeholder = Input(shape=(dims_tfidf,),dtype = "float32")


article_embedded_sequence = article_embedding_layer(article_seq_placeholder)

lstm_headline = Bidirectional(LSTM(num_hidden_units_LSTM))
lstm_article = Bidirectional(LSTM(num_hidden_units_LSTM))

hidden_rep_headline = Dropout(.2)(lstm_headline(headline_embedded_sequence))
hidden_rep_article = Dropout(.2)(lstm_article(article_embedded_sequence))

concat_rep = concatenate([hidden_rep_headline, hidden_rep_article, tfidf_placeholder], axis = -1) #last axis
concat_rep = Dropout(.2)(Dense(64, activation = "relu")(concat_rep))

predictions = Dense(4, activation="softmax")(concat_rep)

bilstm_model = Model(inputs = [headline_seq_placeholder, article_seq_placeholder, tfidf_placeholder], outputs = predictions)
bilstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
bilstm_model.fit([X_train_headline_sequences, X_train_article_sequences, concatenated_train_tfidf], train_labels, validation_data = ([X_test_headline_sequences, X_test_article_sequences, concatenated_test_tfidf], test_labels), epochs = 10)

model_json = bilstm_model.to_json()
with open("final_tfidf_bilstm.json",'w') as json_file:
    json_file.write(model_json)
bilstm_model.save_weights("final_tfidf_bilstm.h5")
print("Saved model: {}".format("final_tfidf_bilstm.json"))
