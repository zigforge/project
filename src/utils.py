from sklearn.preprocessing import (StandardScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder)
from src.configs import Configs as configs, MLModelConfigs
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import sqlite3


def split_data(x_data, y_data):
        X_train, X_test, Y_train, Y_test= \
                        train_test_split(x_data, y_data, test_size=configs.test_size, random_state=123)

        return X_train, Y_train, X_test, Y_test

class DataReader:
    ''' Class to read data from different sources '''
    
    @classmethod
    def read_db(cls, db_file, table, _type=''):
        ''' read data from sqlite db file 
            @param: db_file    -> path to database file
            @param: table      -> name of table to read
            @param: _type      -> type of object to return ['df':pandas dataframe, 'np':numpy array, '': list of tuples]
            
        '''
        
        connection = sqlite3.connect(db_file)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM score")
        data = cursor.fetchall()
        if _type == 'df': 
            columns =  [d[0] for d in cursor.description ]
            data = pd.DataFrame(data, columns=columns)
        
        elif _type == 'np':
            data = np.array(data)
            
        return data

class DataProcessor():
    

    @classmethod
    def process(cls, dataframe, for_classifier=False):
        processed_df = cls.preprocess_df(dataframe)
        processed_df = cls.standardize_df(processed_df)

        y_data = list(processed_df[configs.target_column])
        processed_df.drop(configs.target_column, axis=1, inplace=True)
        x_data  = processed_df

        if for_classifier:
            if MLModelConfigs.cls_type == 'Binary':
                y_data = [cls.get_binary_bin(value) for value in y_data]
            
            else:
                y_data = [cls.get_multi_class_bin(value) for value in y_data]

        x_train, y_train, x_test, y_test = split_data(x_data, y_data)  
        x_train, x_test = cls.encode(x_train, x_test) 
        print('data processing done...')
        return x_train, y_train, x_test, y_test

    @classmethod
    def get_binary_bin(cls, value):
        ''' 0:fail, 1:pass'''

        if value in range(0, 60): 
            return 0
        
        else:   
            return 1

    @classmethod
    def get_multi_class_bin(cls, value):
        '''0:fail, 1:average, 2:excellent'''
        
        if value in range(0, 51): 
            return 0
        
        elif value in range(51, 70):
            return 1
        
        else:   
            return 2
  
    @classmethod
    def preprocess_df(cls, df):
        df.drop('index', axis=1, inplace=True)
        print('removing duplicate records...')
        df.drop_duplicates('student_id', keep='first', inplace=True)
        df.drop('student_id', axis=1, inplace=True)

        # drop row where 'final_test' is null
        df = df[df['final_test'].notna()]

        # impute 'attendance_rate' by average 'attendance_rate'
        avg_attendance_rate = df['attendance_rate'].mean()
        df['attendance_rate'].fillna(avg_attendance_rate,inplace=True)

        # replacing sleep_time & wake_time by sleep_hours
        df['sleep_hours'] = pd.to_datetime(df['wake_time']) - pd.to_datetime(df['sleep_time'])
        df['sleep_hours'] = df['sleep_hours'].apply(lambda x: x.seconds/3600)
        
        # add late_sleep tag for sleep_time > 5am
        df['late_sleep'] = "No"
        df.loc[pd.to_datetime(df['sleep_time'])<pd.to_datetime("05:00"),'late_sleep']="Yes"
        
        df.drop('sleep_time', axis=1, inplace=True)
        df.drop('wake_time', axis=1, inplace=True)

        return df

    @classmethod
    def standardize_df(cls, df):
        cat_cols = ['direct_admission', 'CCA', 'learning_style', 'gender', 'tuition', 'mode_of_transport', 'bag_color', 'late_sleep']
        for col in cat_cols:
            # change values of columns to lower case
            df[col] = df[col].str.lower()

        df.loc[df['tuition']=="y", 'tuition']='yes'
        df.loc[df['tuition']=="n", 'tuition']='no'

        # clean age columns
        df.age.replace(6, 16, inplace=True)
        df.age.replace(5, 15, inplace=True)
        df = df[df.age > 0]

        return df
    
    @classmethod
    def extract_categorical_columns(cls, dataset):
        '''extract categorical columns from pandas dataframe'''
        
        data_type = ['object', 'category']
        cat_cols = []
        
        for i in data_type:
            cols = dataset.dtypes[dataset.dtypes==i]
            cols = list(cols.index)
            cat_cols += cols
        return cat_cols

    @classmethod
    def encode_labels_le(cls, train_data, test_data, categorical_cols):
        le = LabelEncoder()
        for col in categorical_cols:
            train_data.loc[:, col] = le.fit_transform(train_data.loc[:, col])
            test_data.loc[:, col] = le.transform(test_data.loc[:, col])
        
        return train_data, test_data

    @classmethod
    def encode_labels_onehot(cls, train_data, test_data, categorical_cols):
        oe = OneHotEncoder(drop='first')
        ct = ColumnTransformer([('one_hot_encoder', oe, categorical_cols)], 
                                                remainder = 'passthrough')
        train_data = ct.fit_transform(train_data)
        test_data = ct.transform(test_data)
        return train_data, test_data

    @classmethod
    def encode(cls, train_data, test_data):
        print('encoding categorical labels...')
        categorical_cols = cls.extract_categorical_columns(train_data)
        
        if configs.encoder == 'LabelEncoder':
            train_data, test_data = cls.encode_labels_le(train_data, test_data, categorical_cols)
        
        elif configs.encoder == 'OneHotEncoder':
            train_data, test_data = cls.encode_labels_onehot(train_data, test_data, categorical_cols)
        
        else:
            print(f"incorrect encoder '{configs.encoder}' ")
        
        return train_data, test_data
