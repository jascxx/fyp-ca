from features_info import CATEGORICAL_FEATURES, COLUMN_NAMES
import pandas as pd
from numpy import inf, nan

class Preprocessor:
    
    def __init__(self, df):
        df.columns = COLUMN_NAMES
        self.source_df = df
        self.original_columns = None
        self.bounds = None
        self.preprocessed_columns = None

    def get_preprocessed(self, normal_only=False):
        '''
        Get preprocessed NSL-KDD dataframe. This function
        - separates the features from the labels.
        - converts the categorical columns into one hot encoding.
        - apply min-max normalization to the features.
    
        Params:
        - normal_only: True if only the normal data is needed

        Returns:
        - X                     : The preprocessed features.
        - y                     : The labels.
        '''

        if normal_only:
            df = self.source_df[self.source_df['class'] == 'normal'] # only use normal records
        else:
            df = self.source_df.copy()
            
        y = df['class'].map(lambda x : 0 if x == 'normal' else 1)
        X = df.drop(['class', 'difficulty'], axis=1)
        
        
        self.original_columns = X.columns

        X = self._to_one_hot(X)
        min_bound, max_bound = X.min(), X.max()
        X = (X - min_bound)/(max_bound - min_bound)
        # NaN values may occur because train_X.max() - train_X.min() == 0, 
        # that is all values in the column are the same. 
        # For simplicity, we replace the NaN values with its original value = 0.
        # We do not drop these columns as we want to keep the 122 features for consistency.
        X = X.fillna(0)

        self.preprocessed_columns = X.columns
        self.bounds = (min_bound, max_bound)
        return X, y

    def preprocess_another(self, df):
        '''
        Preprocess NSL-KDD dataframe the same way as the `get_preprocessed` function.
        But, the instead of using the min & max values of the dataframe to apply
        min-max normalization, we use the parameters of `source_df` to normalize the 
        dataframe, and then the dataframe is clipped to lie in the interval [0,1].
        '''
        if self.bounds is None:
            raise Exception('self.get_preprocessed must be called before calling self.preprocess_another')
        min_bound, max_bound = self.bounds
        df.columns = COLUMN_NAMES
        
        if 'class' in df.columns:
            y = df['class'].map(lambda x : 0 if x == 'normal' else 1)
            X = df.drop(['class', 'difficulty'], axis=1)
        else:
            y = None
            X = df.copy()
            
        X = self._to_one_hot(X)
        X = (X - min_bound) / (max_bound - min_bound)
        # features that are all the same in the training set may result in inf or -inf values
        # in the testing set, we replace this with 0 as these features do not contribute anything.
        X = X.replace(inf, nan).replace(-inf, nan)
        X = X.fillna(0)
        
        # make sure column order is consistent
        X = X[self.preprocessed_columns]

        return X, y
    
    def reverse_preprocess(self, X_tensor):
        '''
        Reverse the preprocessing function on tensor X.
        
        Returns:
        - X_df          : A dataframe with the original 41 features.
        '''
        min_bound, max_bound = self.bounds
        
        X_df = pd.DataFrame(X_tensor.numpy())
        X_df.columns = self.preprocessed_columns
        X_df = X_df * (max_bound - min_bound) + min_bound
        
        for col in CATEGORICAL_FEATURES:
            X_df[col] = X_df.apply(lambda x: [c for c in CATEGORICAL_FEATURES[col] if x[c] == 1.0][0], axis=1)
        
        return X_df[self.original_columns]
        
    
    def _to_one_hot(self, df):
        one_hot_cols = {}
        for col in CATEGORICAL_FEATURES:
            one_hot_cols = []
            for val in CATEGORICAL_FEATURES[col]:
                one_hot_col = df[col].map(lambda x: 1 if x == val else 0)
                one_hot_col.rename(val)
                one_hot_cols.append(one_hot_col)
            one_hot_cols_df = pd.concat(one_hot_cols, axis=1)
            one_hot_cols_df.columns = CATEGORICAL_FEATURES[col]
            df.drop([col], axis=1, inplace=True)
            df = pd.concat([df, one_hot_cols_df], axis=1)
        return df
