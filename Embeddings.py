import tensorflow as tf 
import tensorflow_hub as hub

import numpy as np 
from tqdm import tqdm

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