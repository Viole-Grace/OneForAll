import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression

class AutoSelect:
    
    def __init__(self, df, features, target, train_split = None):
        """
            Enter a DataFrame with n different Features
            features - list of column names to be used as features for your dataset | Type = <list>
            target - target variable in the dataset | Type = <str>
            train_split - size of the dataset to consider features for.
                            If the entire dataset is used, subsequent modeling might overfit on testing / validation data
                            
                            Default training_split is 0.8 | Type = <int>
            
            SelectKBestFeatures() - Control your model by setting K manually for feature selection. 
                                    (Recommended if you have knowledge in the domain)
                                    
            SelectBestFeatures() - Let the model approximate n top features using regularization, automatically.
                                    (Recommended if you have no knowlege of the domain or are used to AutoML based working)
        """
        if train_split == None:
            train_split = 0.8
            
        self.df = df
        self.train_split = train_split
        
        self.train = df[ : int(len(df)*self.train_split)]
        self.train = self.train.replace(np.nan, 0)
        
        self.test = df[int(len(df)*self.train_split) : ]
        self.test = self.test.replace(np.nan, 0)
        
        self.features = self.train[features]
        self.target = self.train[target]
        self.track = len(features)
        
        self.k_selected_columns = None
        self.top_selected_columns = None
        
    
    def SelectKBestFeatures(self, k):
        
        """
            k = Number of best features you want from the model
        """
        if k > self.track:
            print('Error, Number of features to be selected are more than the number of features in the dataset')
            return
        
        selector = SelectKBest(f_classif, k=k)
        self.features_new = selector.fit_transform(self.features, self.target)
        
        selected_features = pd.DataFrame(selector.inverse_transform(self.features_new),
                                        index=self.train.index,
                                        columns=self.features.columns)
        self.k_selected_columns = selected_features.columns[selected_features.var() != 0]
        self.k_selected_columns = self.train[self.k_selected_columns]
        
        print('Top {} features : \n{}'.format(k, self.k_selected_columns.columns))
        return self.k_selected_columns
    
    def SelectBestFeatures(self):
        
        """
            Approximates best features for the model using L1 regularization
        """
        
        logreg = LogisticRegression(C=1, penalty = 'l1', solver = 'liblinear', random_state=42).fit(self.features, self.target)
        model = SelectFromModel(logreg, prefit=True)
        
        self.features_new = model.transform(self.features)
        
        selected_features = pd.DataFrame(model.inverse_transform(self.features_new),
                                        index=self.features.index,
                                        columns=self.features.columns
                                        )
        self.top_selected_columns = selected_features.columns[selected_features.var() != 0]
        self.top_selected_columns = self.train[self.top_selected_columns]
        
        print('Best selected columns :\n{}'.format(self.top_selected_columns.columns))
        return self.top_selected_columns