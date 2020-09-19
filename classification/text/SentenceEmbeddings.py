import tensorflow as tf 
import tensorflow_hub as hub

import numpy as np 
from tqdm import tqdm

from gensim.models import Word2Vec, FastText

import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

class ELMoVectors:

	def __init__(self, data, batch_size=None):

		if batch_size == None:
			batch_size = 20

		self.data = data
		self.url = 'https://tfhub.dev/google/elmo/2'
		self.embed = hub.Module(self.url)
		self.batch_size = batch_size
		self.batches = None

	def batch_data(self):

		self.batches = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]

	def form_elmo_vectors(self, batch):

		embeddings = self.embed(batch, signature='default', as_dict=True)['default']

		with tf.Session() as sess:

			sess.run(tf.global_variables_initializer())
			sess.run(tf.tables_initializer())
			x = sess.run(embeddings)

			return x

	def get_all_vectors(self):

		print('Number of batches : {}'.format(len(self.data)//self.batch_size))

		self.batch_data()
		print('Making sentence embeddings ...')

		vectors = [tqdm(self.form_elmo_vectors(batch)) for batch in self.batches]
		#flatten them out to a single list of vectors
		embeddings = [item for sublist in vectors for item in sublist]
		# embeddings = np.concatenate([[], vectors])

		return embeddings


"""

USAGE : params -

		 text		| Type : <list> | List of processed sentences to make vectors
		 batch_size | Type : <int>  | Batch size for embeddings to be formed. Embeddings are 1024D.

text = ['This is so cool!','I am pretty excited about the movie','This movie is great!']
elmo = ELMoVectors(text, batch_size=8)
embeddings = elmo.get_all_vectors() #get all vectors

_____________________________________________________

Using ELMoVectors for a Machine Learning model
_____________________________________________________


from sklearn.model_selection import train_test_split
from Embeddings import ELMoVectors

x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.3, random_state=42)


train, test = ELMoVectors(x_tr, batch_size=8), ELMoVectors(x_te, batch_size=8)

train_embeddings = train.get_all_vectors()
test_embeddings = test.get_all_vectors()

train_embeddings, test_embeddings = np.array(train_embeddings), np.array(test_embeddings)

#now just use classifiers as you would normally do

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

svm = SVC()
svm.fit(train_embeddings, y_tr)
y_pred = svm.predict(test_embeddings)

print('Accuracy : {}'.format(accuracy_score(y_pred, y_te)))
print('Precision : {}\nRecall : {}'.format(precision_score(y_pred, y_te), recall_score(y_pred, y_te)))

"""

class Word2VecVectors:
    
    def __init__(self, data):
        
        self.data = data
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        self.features = None
        self.model = None
    
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
        
    def make_model(self):
        
        self.model = Word2Vec(self.features, size=100, workers=4, window=10, iter=20)
        print('Word2Vec model created.')
        
    def form_sentence_vector(self, sent):
        
        vector = []
        
        for word in sent.split():
            try:
                vector.append(self.model[word])
            except:
                vector.append([0]*100)
        
        vec = np.array(vector).mean(axis=0) #average vector of the sentence
        
        return vec
        
    def form_sentence_vectors(self, sentences):
        
        vectors = [self.form_sentence_vector(sent) for sent in sentences]
        
        return vectors
    
"""
USAGE : params -

		 text		| Type : <list> | List of processed sentences to make vectors
    
data = ['This is a sentence for word2vec','This is another one that relates to the same topic','Yet another one for the same purpose','The models are performing well',
       'This is all for a simple test function','Is this what we wanted to try with word2vec?','This could potentially be useful to the opensource community at large']
w2v = Word2VecVectors(data=data)
w2v.process_data()
w2v.make_model()
single_sentence_vector = w2v.form_sentence_vector('This is so cool!')
sentence_vectors = w2v.form_sentence_vectors(['This is so cool!','I am pretty excited about the movie','This movie is great!'])

___________________________________________________

Using Word2VecVectors for a Machine Learning Model
___________________________________________________

from sklearn.model_selection import train_test_split
from Embeddings import Word2VecVectors

w2v_embeddings = Word2VecVectors(data=X) #X is a list of sentences
w2v_embeddings.process_data()
w2v_embeddings.make_model()

x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.3, random_state=42)

train_embeddings = w2v_embeddings.form_sentence_vectors(x_tr)
test_embeddings = w2v_embeddings.form_sentence_vectors(x_te)

train_embeddings, test_embeddings = np.array(train_embeddings), np.array(test_embeddings)

#now just use classifiers as you would normally do

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

svm = SVC()
svm.fit(train_embeddings, y_tr)
y_pred = svm.predict(test_embeddings)

print('Accuracy : {}'.format(accuracy_score(y_pred, y_te)))
print('Precision : {}\nRecall : {}'.format(precision_score(y_pred, y_te), recall_score(y_pred, y_te)))
"""

class FastTextVectors:
    
    def __init__(self,data):
        
        self.data = data
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = word_tokenize
        self.features = None
        self.model = None
    
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
        
    def make_model(self):
        
        self.model = FastText(sentences=self.features, size=100, workers=4, window=10, iter=20, min_count=5, sg=0)
        print('FastText model created. Save with a .bin extension, use gensim.models.FastText.load("<model.bin>") to load the model')
        
    def form_sentence_vector(self, sent):
        
        vector = []
        
        for word in sent.split():
            try:
                vector.append(self.model[word])
            except:
                vector.append([0]*100)
        
        vec = np.array(vector).mean(axis=0) #average vector of the sentence
        
        return vec
        
    def form_sentence_vectors(self, sentences):
        
        vectors = [self.form_sentence_vector(sent) for sent in sentences]
        
        return vectors
        
"""
USAGE : params -

		 text		| Type : <list> | List of processed sentences to make vectors
    
data = ['This is a sentence for word2vec','This is another one that relates to the same topic','Yet another one for the same purpose','The models are performing well',
       'This is all for a simple test function','Is this what we wanted to try with word2vec?','This could potentially be useful to the opensource community at large']
fast = FastTextVectors(data=data)
fast.process_data()
fast.make_model() #time intensive operation
single_sentence_vector = fast.form_sentence_vector('This is so cool!')
sentence_vectors = fast.form_sentence_vectors(['This is so cool!','I am pretty excited about the movie','This movie is great!'])

___________________________________________________

Using FastTextVectors for a Machine Learning Model
___________________________________________________

from sklearn.model_selection import train_test_split
from Embeddings import FastTextVectors

ft_embeddings = FastTextVectors(data=X) #X is a list of sentences
ft_embeddings.process_data()
ft_embeddings.make_model()

x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.3, random_state=42)

train_embeddings = ft_embeddings.form_sentence_vectors(x_tr)
test_embeddings = ft_embeddings.form_sentence_vectors(x_te)

train_embeddings, test_embeddings = np.array(train_embeddings), np.array(test_embeddings)

#now just use classifiers as you would normally do

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

svm = SVC()
svm.fit(train_embeddings, y_tr)
y_pred = svm.predict(test_embeddings)

print('Accuracy : {}'.format(accuracy_score(y_pred, y_te)))
print('Precision : {}\nRecall : {}'.format(precision_score(y_pred, y_te), recall_score(y_pred, y_te)))
"""