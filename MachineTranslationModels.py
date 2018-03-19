# Importing data
#Importing data
import helper
# Load English data
english_sentences = helper.load_data('hello.csv')
# Load French data
french_sentences = helper.load_data('french.csv')

print('Dataset Loaded')

# Tokenization

import project_tests as tests
from keras.preprocessing.text import Tokenizer

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    
    t = Tokenizer()
    t.fit_on_texts(x)
    tokens = t.texts_to_sequences(x)

    return tokens, t

tests.test_tokenize(tokenize)

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']

text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
    
# preprocessing

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)

print('Data Preprocessed')

# Vanilla RNN

import numpy as np
from test import *
from datetime import datetime
import sys
import os
import time



def sigmoid(net):
    return 1.0/(1.0+np.exp(-net))

def helper(v):
    return v*(1-v)

def softmax(x):
    
    value = np.exp(x - np.max(x))
    return value / value.sum()


class TranslatorRNN(object):
    
    def __init__(self, inputDim, targetDim, hiddenDim, backPropogate_truncate=5):
       
        self.inputDim = inputDim
        self.targetDim = targetDim
        self.hiddenDim = hiddenDim
        self.backPropogate_truncate = backPropogate_truncate
        
        self.We = np.random.uniform(-np.sqrt(1./inputDim), np.sqrt(1./inputDim), (hiddenDim, inputDim))
        self.Ue = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim, hiddenDim))
        self.Wd = np.random.uniform(-np.sqrt(1./targetDim), np.sqrt(1./targetDim), (hiddenDim, targetDim))
        self.Ud = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim, hiddenDim))
        self.V = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim, hiddenDim))
        self.P = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (targetDim,hiddenDim))

    def loss(self, x, y):
        length = len(y)
        h = self.encode(x)
        c = h[-1]
        s,o = self.decode(x, c, y)
        target_probability = o[np.arange(length), y]
        L = -1 *np.sum(np.log(target_probability))
        return L

    def totalLoss(self, x, y):
       
        L = 0
        N = 0
        for i in np.arange(len(y)):
            L += self.loss(x[i],y[i])
            N += len(y[i])
        return L / N

    def encode(self, x):
        
        lengthX = len(x)
        
        h = np.zeros((lengthX+1, self.hiddenDim))
        
        for t in np.arange(lengthX):
            
            h[t] = sigmoid(self.We[:, x[t]] + np.dot(self.Ue, h[t-1]))
        return h

    def decode(self, x, c, y=None):
        
        if (y != None):
            
            length = len(y)
            
            s = np.zeros((length, self.hiddenDim))
            s[0] = sigmoid(np.dot(self.V, c))
            o = np.zeros((length, self.targetDim))
            o[0] = softmax(np.dot(self.P, s[0]))
            # For each time step...
            for t in np.arange(1,length):
                
                s[t] = sigmoid(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c))
                o[t] = softmax(np.dot(self.P, s[t]))
            return [s, o]
        
        else:
            
            s = [sigmoid(np.dot(self.V, c))]
            o = [softmax(np.dot(self.P, s[0]))]
            y = [np.argmax(o)]
            t = 1
            
            while y[-1] != word2vec_target[sentence_end_token] and len(y)<30 :
                s.append(sigmoid(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c)))
                o.append(softmax(np.dot(self.P, s[t])))
                
                trial = word2vec_target[unknown_token]
                while(trial == word2vec_target[unknown_token]):
                    samples = np.random.multinomial(1, o[t])
                    trial = np.argmax(samples)
                y.append(trial)
                t = t + 1
            y = np.asarray(y)
            o = np.asarray(o)
            s = np.asarray(s)
        return [s, o, y]

    def translate(self, x):
        target=[]
        for i in range(len(x)):
            h = self.encode(x[i])
            s, o, y = self.decode(x[i], h[-1])
            target.append([vec2word_target[w] for w in y[:-1]])
        return target

    def backPropogate(self, x, y):
        lengthX = len(x)
        length = len(y)
       
        h = self.encode(x)
        c = h[-1]
        s,o = self.decode(x, c, y)
        
        dLdUe = np.zeros(self.Ue.shape)
        dLdWe = np.zeros(self.We.shape)
        dLdP = np.zeros(self.P.shape)
        dLdV = np.zeros(self.V.shape)
        dLdUd = np.zeros(self.Ud.shape)
        dLdWd = np.zeros(self.Wd.shape)
        delta_c_t = helper(c)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        #check the gradients and weight
        
        delta_o=np.asarray([weight*delta_o[t] for t in np.arange(len(y))])
        
        for t in np.arange(length)[::-1]:
            #check for the calculations
            dLdP += np.outer(delta_o[t], s[t])
            #print the values
            delta_s_t = np.dot(self.P.T, delta_o[t]) * helper(s[t])
            dLdc = np.zeros(self.hiddenDim)
            
            for decode_step in np.arange(max(1, t - self.backPropogate_truncate), t + 1)[::-1]:
                
                # summation of gradiemts to the previous steps
                dLdUd += np.outer(delta_s_t, s[decode_step - 1])
                dLdV += np.outer(delta_s_t, c)
                dLdWd[:, y[decode_step - 1]] += delta_s_t
                dLdc += np.dot(self.V.T, delta_s_t)
                
                delta_s_t = np.dot(self.Ud.T, delta_s_t) * helper(s[decode_step - 1])
            if t-self.backPropogate_truncate < 1 :
                dLdV += np.outer(delta_s_t, c)
                dLdc += np.dot(self.V.T, delta_s_t)
            delta_c_t = dLdc * helper(c)
            for encode_step in np.arange(max(0, lengthX - self.backPropogate_truncate), lengthX)[::-1]:
                dLdWe[:, x[encode_step]] += delta_c_t
                dLdUe += np.outer(delta_c_t, h[encode_step - 1])
                
                delta_c_t = np.dot(self.Ue.T, delta_c_t) * helper(h[encode_step - 1])

        return [dLdUe, dLdWe, dLdUd, dLdWd, dLdV, dLdP]


    def SGD(self,x,y,eta):
        dLdUe, dLdWe, dLdUd, dLdWd, dLdV, dLdP = self.backPropogate(x,y)
        self.Ue -= eta * dLdUe
        self.We -= eta * dLdWe
        self.Ud -= eta * dLdUd
        self.Wd -= eta * dLdWd
        self.V -= eta* dLdV
        self.P -= eta* dLdP

    def train(model, x, y, learning_rate=0.01, nepoch=10, evaluate_loss_after=1):
        
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            
            if epoch % evaluate_loss_after == 0:
                loss = model.totalLoss(x,y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print '%s: error after #examples=%d epoch=%d: %f' % (time, num_examples_seen, epoch, loss)
                
                
           
            for i in range(len(y)):
                
                model.SGD(x[i], y[i], learning_rate)
                num_examples_seen += 1
        return losses

   


read_en, read_fr = load_dataset('hello.csv','french.csv')

train_en, word2vec_resource, vec2word_resource, vocab_source, sentence_freq_source= read_en[0], \
    read_en[1], read_en[2], read_en[3], read_en[4]
train_fr, word2vec_target, vec2word_target, vocab_target, sentence_freq_target = read_fr[0], \
    read_fr[1], read_fr[2], read_fr[3], read_fr[4]


sentence_freq_target = [x[1] for x in sentence_freq_target]
sentence_freq_target.append(1)
sentence_freq_target=np.asarray(sentence_freq_target)
temp = np.log((sentence_freq_target.astype(float)+1))
weight = temp+1

model1 = TranslatorRNN(vocabulary_size,vocabulary_size,50)

temp= ["my name is Anmol Khanna"]
model1.train(train_en[:3000],train_fr[:3000])

trial1 = model1.translate(train_en[10:12])
#trial1 = model1.translate(temp);
print len(trial1)
for i in range(len(trial1)):
    print " ".join([vec2word_resource[x] for x in train_en[10+i][:-1]])
    print " ".join(trial1[i])
    print " ".join([vec2word_target[x] for x in train_fr[10+i][:-1]])
    
# Simple RNN

from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.layers import Activation
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.layers import Bidirectional

def simple_model(input, output_size, english_size, french_size):
    """
    Build and train a basic RNN on x and y
    :param input: Tuple of input shape
    :param output_size: Length of output sequence
    :param english_size: Number of unique English words in the dataset
    :param french_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    
    model = Sequential()
    model.add(GRU(output_size, return_sequences=True, input=input[1:]))
    
    #print(model.output_shape)
    model.add(TimeDistributed(Dense(french_size, activation='softmax')))
    #print(model.output_shape)
    
    learning_rate = 0.01
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_simple_model(simple_model)


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index),
    len(french_tokenizer.word_index))

print(simple_rnn_model.summary())

simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))

# Bidirectional RNN

def bd_model(input, output_size, english_size, french_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input: Tuple of input shape
    :param output_size: Length of output sequence
    :param english_size: Number of unique English words in the dataset
    :param french_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement

    model = Sequential()
    model.add(Bidirectional(GRU(output_size, return_sequences=True), input=input[1:]))
    model.add(TimeDistributed(Dense(french_size, activation="softmax")))
    
    learning_rate = 0.01
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
tests.test_bd_model(bd_model)


# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
bd_model_ = bd_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index),
    len(french_tokenizer.word_index))

print(bd_model_.summary())

bd_model_.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

# Print prediction(s)
print(logits_to_text(bd_model_.predict(tmp_x[:1])[0], french_tokenizer))
