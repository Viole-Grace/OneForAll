from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import numpy as np

import re

import gc

from gensim.models import Word2Vec, FastText

class Word2VecEmbeddings:
    
    def __init__(self,data):
        
        self.data = data
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        self.features = None
        self.text_vocab = None
        self.model = None
        self.dim = None
        self.pretrained_weights = None
    
    def process_data(self, lemmatize=False, stem=False, strip_spaces=True, remove_special_chars=True, lower=True):
        
        if lower == True:
            self.data = [sent.lower() for sent in self.data]
        
        if remove_special_chars == True:
            self.data = [re.sub('[^A-Za-z0-9 ]+', '', sent) for sent in self.data]
        
        if strip_spaces == True:
            self.data = [re.sub(' +',' ',sent).strip() for sent in self.data]
            
        if stem == True:
            self.data = [' '.join(self.stemmer.stem(word) for word in self.tokenizer(sent)) for sent in self.data]
            
        if lemmatize == True:
            self.data = [' '.join(self.lemmatizer.lemmatize(word) for word in self.tokenizer(sent)) for sent in self.data]
            
        self.features = [word for word in [sent.split() for sent in self.data]]
        
        print('Processed Data\n')
        
    def make_model(self, dim=None, win=None, iter=None):
        
        if dim == None:
            dim = 100
        if win == None:
            win = 10
        if iter == None:
            iter = 20
        
        #check if I am right - word vocab and model vocab can be different
        word_dict = {}
        for sent_list in self.features:
            for word in sent_list:
                if word in word_dict:
                    word_dict[word] +=1
                else:
                    word_dict[word] = 1
        self.text_vocab = len(word_dict)
        print('Text vocab : {}'.format(self.text_vocab))
        
        del word_dict
        gc.collect()
        
        self.dim = dim
        self.model = Word2Vec(self.features, size=dim, workers=4, window=win, iter=iter, min_count=10)
        print('Model vocab : {}\n'.format(len(self.model.wv.vocab)))
        print('{}D Word2Vec model created. Load using gensim.models.Word2Vec.load(<model name>.model)'.format(self.dim))
        
    def get_embeddings(self):
        
        embedding_matrix = np.zeros((self.text_vocab+1, self.dim))
        for i in range(len(self.model.wv.vocab)):
            embedding_vector = self.model.wv[self.model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        self.pretrained_weights = embedding_matrix
        
#         self.pretrained_weights = self.model.wv.syn0
        vocab_size, embedding_size = self.pretrained_weights.shape
        
        print('Returned : pretrained_weights, vocab_size and embedding_size\n')
        print('These weights can be passed into the Embedding layer of a neural network. A neural network can usually have only one Embedding Layer.\n')
        print('Using the Keras.Sequential() API :\n\tEmbedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))')
        
        return self.pretrained_weights, vocab_size, embedding_size
    

class FastTextEmbeddings:
    
    def __init__(self,data):
        
        self.data = data
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        self.features = None
        self.text_vocab = None
        self.model = None
        self.dim = None
        self.pretrained_weights = None
    
    def process_data(self, lemmatize=False, stem=False, strip_spaces=True, remove_special_chars=True, lower=True):
        
        if lower == True:
            self.data = [sent.lower() for sent in self.data]
        
        if remove_special_chars == True:
            self.data = [re.sub('[^A-Za-z0-9 ]+', '', sent) for sent in self.data]
        
        if strip_spaces == True:
            self.data = [re.sub(' +',' ',sent).strip() for sent in self.data]
            
        if stem == True:
            self.data = [' '.join(self.stemmer.stem(word) for word in self.tokenizer(sent)) for sent in self.data]
            
        if lemmatize == True:
            self.data = [' '.join(self.lemmatizer.lemmatize(word) for word in self.tokenizer(sent)) for sent in self.data]
            
        self.features = [word for word in [sent.split() for sent in self.data]]
        
        print('Processed Data\n')
        
    def make_model(self, dim=None, win=None, iter=None, sg=None):
        
        if dim == None:
            dim = 100
        if win == None:
            win = 10
        if iter == None:
            iter = 20
        if sg == None:
            sg = 0
            
        #check if I am right - word vocab and model vocab can be different
        word_dict = {}
        for sent_list in self.features:
            for word in sent_list:
                if word in word_dict:
                    word_dict[word] +=1
                else:
                    word_dict[word] = 1
        self.text_vocab = len(word_dict)
        print('Text vocab : {}'.format(self.text_vocab))
        
        del word_dict
        gc.collect()
        
        self.dim = dim
        self.model = FastText(sentences=self.features, size=dim, workers=4, window=win, iter=iter, min_count=10, sg=sg)
        print('Model vocab : {}\n'.format(len(self.model.wv.vocab)))
        print('{}D FastText model created. Save with a .bin extension, use gensim.models.FastText.load("<model.bin>") to load the model'.format(dim))
        
    def get_embeddings(self):
        
        embedding_matrix = np.zeros((self.text_vocab+1, self.dim))
        for i in range(len(self.model.wv.vocab)):
            embedding_vector = self.model.wv[self.model.wv.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        self.pretrained_weights = embedding_matrix
#         self.pretrained_weights = self.model.wv.syn0

        vocab_size, embedding_size = self.pretrained_weights.shape
        
        print('Returned : pretrained_weights, vocab_size and embedding_size\n')
        print('These weights can be passed into the Embedding layer of a neural network. A neural network can usually have only one Embedding Layer.\n')
        print('Using the Keras.Sequential() API :\n\tEmbedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))')
        
        return self.pretrained_weights, vocab_size, embedding_size