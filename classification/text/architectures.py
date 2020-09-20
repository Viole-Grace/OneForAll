from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input ,Dense, GRU, Activation, Dropout, Embedding, Bidirectional, LSTM
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, concatenate

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from statistics import median
import pickle

#Binary, MultiClass, MultiLabel -> All three should be made

class Binary:
    
    def __init__(self, data, targets, nodes=None, dropout=None, pretrained=False):
        
        if nodes == None:
            nodes = 128
        if dropout == None:
            dropout = 0.25
            
        self.data = data
        self.targets = targets
        self.pretrained = pretrained
        self.nodes = nodes
        self.dropout = dropout
        self.tokenizer = None
        self.average_length = None
        self.vocab_size = None
        self.features = None
        self.model = None
        self.batch_size = None
        
        print('Model Type : Binary\nPretrained = {}\nNodes per layer = {}\n'.format(self.pretrained, self.nodes))
        
    def clean_data(self, lemmatize=False, stem=False, strip_spaces=True, remove_special_chars=True, lower=True):
        
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
        
        print('Cleaned Data\n')

    def process_data(self):
        
        self.clean_data()
        
        t = Tokenizer()
        t.fit_on_texts(self.data)
        text_mat = t.texts_to_sequences(self.data)
                
        if self.pretrained == False:
            
            self.vocab_size = len(t.word_index)+1
        
        self.average_length = int(median([len(i) for i in self.data]))
        self.features = pad_sequences(text_mat, maxlen=self.average_length, padding='post')
        self.tokenizer = t
        
        if len(self.data) <5000:
            self.batch_size = 16
        elif len(self.data) >=5000 and len(self.data) <10000:
            self.batch_size = 32
        elif len(self.data) >=10000 and len(self.data) <30000:
            self.batch_size = 64
        else:
            self.batch_size = 128
            
        with open('tokenizer.pickle','wb') as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def TrainOneLayerLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '1LayerLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(LSTM(self.nodes, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("1_layer_{}_nodes_LSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
        
        return model
    
    def TrainThreeLayerLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '3LayerLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("3_layer_{}_nodes_LSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
        
        return model
    
    def TrainFiveLayerLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '5LayerLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))
        model.add(LSTM(self.nodes))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("5_layer_{}_nodes_LSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainOneLayerGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '1LayerGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(embed_layer)
        avg_pool = GlobalAveragePooling1D()(gru_layer)
        max_pool = GlobalMaxPool1D()(gru_layer)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("1_layer_{}_nodes_GRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainThreeLayerGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '3LayerGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(embed_layer)
        gru_layer_2 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer)
        gru_layer_3 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer_2) 
        avg_pool = GlobalAveragePooling1D()(gru_layer_3)
        max_pool = GlobalMaxPool1D()(gru_layer_3)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("3_layer_{}_nodes_GRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainFiveLayerGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '5LayerGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(embed_layer)
        gru_layer_2 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer)
        gru_layer_3 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer_2)
        gru_layer_4 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer_3)
        gru_layer_5 = GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)(gru_layer_4)
        avg_pool = GlobalAveragePooling1D()(gru_layer_5)
        max_pool = GlobalMaxPool1D()(gru_layer_5)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("5_layer_{}_nodes_GRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainOneLayerBiLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '1LayerBiLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(Bidirectional(LSTM(self.nodes, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("1_layer_{}_nodes_BiLSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainThreeLayerBiLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '3LayerBiLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes)))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("3_layer_{}_nodes_BiLSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainFiveLayerBiLSTM(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        model = Sequential()
        
        weight_file = '5LayerBiLSTM_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
            
        if self.pretrained == True:
            model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, weights=[pretrained_weights], trainable=False))
        
        else:
            model.add(Embedding(self.vocab_size, self.nodes))
        
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)))
        model.add(Bidirectional(LSTM(self.nodes)))
        model.add(Dense(1, activation='sigmoid'))
        
        print(model.summary())
        
        model.compile(optimizer=Adam(clipnorm=1.), metrics=['accuracy'], loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("5_layer_{}_nodes_BiLSTM_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainOneLayerBiGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '1LayerBiGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(embed_layer)
        avg_pool = GlobalAveragePooling1D()(gru_layer)
        max_pool = GlobalMaxPool1D()(gru_layer)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("1_layer_{}_nodes_BiGRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainThreeLayerBiGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '3LayerBiGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(embed_layer)
        gru_layer_2 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer)
        gru_layer_3 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer_2)
        avg_pool = GlobalAveragePooling1D()(gru_layer_3)
        max_pool = GlobalMaxPool1D()(gru_layer_3)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("3_layer_{}_nodes_BiGRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model
    
    def TrainFiveLayerBiGRU(self, epochs=None, embedding_matrix=None, vocab_size=None, embedding_size=None):
        
        orig_input = Input(shape=self.average_length)
        
        weight_file = '5LayerBiGRU_{}_nodes_weights.hdf5'.format(self.nodes)
        ckpt = ModelCheckpoint(weight_file,
                               save_best_only=True,
                               save_weights_only=True,
                               monitor='val_accuracy')
        if epochs == None:
            epochs = 20
        
        if self.pretrained == True:
            embed_layer = Embedding(input_dim = vocab_size, output_dim = embed_size, weights = [embedding_matrix], trainable=False)(orig_input)
        
        else:
            embed_layer = Embedding(self.vocab_size, self.nodes)(orig_input)
            
        gru_layer = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(embed_layer)
        gru_layer_2 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer)
        gru_layer_3 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer_2)
        gru_layer_4 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer_3)
        gru_layer_5 = Bidirectional(GRU(self.nodes, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout))(gru_layer_4)
        avg_pool = GlobalAveragePooling1D()(gru_layer_5)
        max_pool = GlobalMaxPool1D()(gru_layer_5)

        conc_layer = concatenate([avg_pool, max_pool])
        prediction_layer = Dense(1, activation='sigmoid')(conc_layer)

        model = Model(inputs=orig_input, outputs=prediction_layer)
        
        print(model.summary)
        
        model.compile(optimizer=Adam(clipnorm=1.),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')
        
        print('Trained weights file : {}'.format(weight_file))
        
        model.fit(self.features, self.targets, validation_split=0.25, epochs=epochs, batch_size=self.batch_size, callbacks=[ckpt])
        self.model = model
        model_json = model.to_json()
        with open("5_layer_{}_nodes_BiGRU_model.json".format(self.nodes), "w") as json_file:
            json_file.write(model_json)
            
        return model        
        
    def evaluate_model(self, text, weight_file, tokenizer_file, model_file):
        
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        loaded_model.load_weights(weight_file)
        loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(clipnorm=1.))
        
        tokenizer_ = open(tokenizer_file,'rb')
        tokenizer = pickle.load(tokenizer_)
        tokenizer_.close()
        
        features = tokenizer.texts_to_sequences([text])
        padded_features = pad_sequences(feature, maxlen=self.average_length, padding='post')
        
        labels = loaded_model.predict(padded_features)
        print(lables[0], labels)