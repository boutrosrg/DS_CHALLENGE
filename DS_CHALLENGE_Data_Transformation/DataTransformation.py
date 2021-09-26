# -*- coding: utf-8 -*-

'''
This class apply the following transformations on dataframe

    1- drop columns with high ratio of missing data
    2- get datetime features
    3- get categorical features
    4- encode categorical features into dummy ones
    5- encode datetime features to numerical ones
    6- generate new timediff features out of existing datetime features
    7- delete datetime features after encoding them
    8- impute missing data
    9- drop low-variance features
    10-split labels & features of data
    11-scale data around 0 average (standardization)
'''

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development"  

import operator
import json
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)

class DataTransformation:
    def __init__(self, df: pd.DataFrame, datasettype: str):
        with open("./data_trans_settings.json") as datatrans_settings:

            # read module settings file
            dt_settings = json.load(datatrans_settings) 

            # max ratio of allowed nan values in a columns
            self.maxnanthreshold= dt_settings["max_nan_threshold"]                
            
            # max. number of unique values in categorical features   
            self.maxuniquesfactor = dt_settings["max_uniques_factor"]           
            
            # list of heterogeneous columns (i.e. cols with different data types)
            self.hetcol= dt_settings["het_col"]                                  
            
            # list of targeted date/time features (i.e. "year", "month", ...)
            self.targeteddtfeatures= dt_settings["targeted_dt_features"]
            
            # list of grouping columns to be considered before data imputation
            self.impgroupcol= dt_settings["imp_group_col"]                         
            
            # criteria of data imputation
            self.impcriteria= dt_settings["imp_criteria"]                         
            
            # min ratio of allowed std value in a column
            self.minstdthreshold= dt_settings["min_std_threshold"]                 
            
            # name of target variable
            self.targetvar= dt_settings["target_variable"]                         
            
            # list of indexing (i.e. non-regressors) cols
            self.indexingcols= dt_settings["indexing_cols"]                        
        
        # data + target dataset
        self.df = df
        
        # dataset type (train/test)
        self.datasettype= datasettype

    # init property of dataset type
    datasettype = property(operator.attrgetter('_datasettype'))

    # apply a value check property on dataset type
    @datasettype.setter
    def datasettype(self, dst):
        if dst not in ['train', 'test']:
            raise ValueError("Invalid dataset type. Expected one of: %s" % datafile_types)
        self._datasettype = dst

    # init property of dataset
    df = property(operator.attrgetter('_df'))

    # apply a non-empty check property on dataset
    @df.setter
    def df(self, dataframe1):
        if dataframe1.empty: 
            raise Exception("df cannot be empty")
        self._df = dataframe1

    def drop_low_inf_cols(self)-> pd.DataFrame:
    
        '''
        this function drops columns with high ratio of missing data 
        i.e. above given threshold

        Returns
        -------
        df : pandas df
            dataframe without high missing data 
        '''        
        
        try:
            print("drop columns with high ratio of missing data... ")
            self.df = self.df.loc[:,self.df.isna().mean() <= self.maxnanthreshold]        
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df

    def get_datetime_features(self)-> pd.DataFrame:
    
        '''
        this function changes datetime columns in dataframe to columns of type (datetime) 

        Returns
        -------
        df: pandas df
            data with updated columns data types
        '''                
        
        try:
            # get datetime col names
            print("get datetime features... ")

            for col in self.df.columns:
                if self.df[col].dtype == 'object':            
                    self.df[col] = pd.to_datetime(self.df[col], errors='ignore')
            
            print("done!")
            
        except Exception as e:
            print(e)            
        
        return self.df

    def get_cat_features(self)-> list:
        
        '''
        this function gets list of categorical columns based on max. number 
        of unique values

        Returns
        -------
        cat_list:list
            list of df categorical columns
        '''
                        
        try:
            # get string columns
            print("get categorical features... ")

            self.df= self.df.convert_dtypes(
                infer_objects=True, convert_string=True, 
                convert_integer=True, convert_floating=True)
            
            # set cat features
            cat_list= []    

            # set threshold of max. unique values in cat. features 
            th= int(len(self.df) * self.maxuniquesfactor)
            
            # append to cat_list
            [cat_list.append(f) 
                for f in list(self.df.columns) if self.df[f].nunique() <= th]          
            
            # get string cols
            str_list= list(self.df.columns[self.df.dtypes=='string'])
        
            print("done!")
            
        except Exception as e:
            print(e)
            
        cat_list= list(set (cat_list + str_list))
        
        return cat_list

    def drop_heterogeneous_cols(self, cat_features: list) -> Tuple[pd.DataFrame, list]:
        
        '''
        this function drop cols without unique data type (i.e. 
        heterogeneous columns)

        Returns
        -------
        df: pandas df
            data with updated columns data types
        cat_list:list
            list of df categorical columns
        '''

        try:
            # drop heterogeneous columns from dataframe & cat. list
            print("drop heterogeneous columns... ")
            
            if self.hetcol in self.df.columns:                
                self.df= self.df.drop([self.hetcol], axis=1)
                
            if self.hetcol in cat_features:                
                cat_features.remove(self.hetcol)   

            print("done!")
            
        except Exception as e:
            print(e)

        return self.df, cat_features

    def encode_cat_features(self, cat_features: list)-> pd.DataFrame:    
        
        '''
        this function encode categorical features into numerical (i.e. dummy) 
        columns using pandas get_dummies

        Parameters
        ----------
        cat_features: list
            list of data categorical features

        Returns
        -------
        df : pandas df
            dataframe with encoded categorical features
        '''
                
        try:
            # encode categorical cols
            print("encode categorical features... ")
            self.df = pd.get_dummies(self.df, columns=cat_features)
        
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df   

    def encode_datetime_features(self)-> pd.DataFrame:
    
        '''
        this function encodes datetime features into numerical columns 

        Returns
        -------
        df : pandas df
            dataframe of encoded datetime features
        '''                

        try:
            # get datetime features list
            dt_list= list(self.df.columns[self.df.dtypes=='datetime64[ns]'])            

            # set columns names to lowercase
            print("encode datetime features... ")

            self.df.columns= self.df.columns.str.lower()
            
            # remove spaces from columns names
            self.df.columns = self.df.columns.str.replace(' ', '_') 

            # generate numerical features out of datetime features
            for f in dt_list:
                for t in self.targeteddtfeatures:                           
                    self.df[f + '_' + t] = getattr(self.df[f].dt, t)                                     

            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df
        
    def add_new_timediff_features(self)-> pd.DataFrame:
        
        '''
        this function adds new time difference feetures between existing
        datatime ones

        Returns
        -------
        df : pandas df
            dataframe with new custom features calculated from datetime ones
        '''
                
        try:
            # generate time difference attributes
            print("generate new timediff features... ")

            self.df["expected_start_start_process_diff"] = \
            (self.df['start_process']-self.df['expected_start']).astype('timedelta64[m]')
            
            self.df["start_process_process_end_diff"] = \
            (self.df['process_end']-self.df['start_process']).astype('timedelta64[m]')
            
            self.df["start_subprocess1_subprocess1_end_diff"] = \
            (self.df['subprocess1_end']-self.df['start_subprocess1']).astype('timedelta64[m]')
            
            self.df["start_critical_subprocess1_subprocess1_end_diff"] = \
            (self.df['subprocess1_end']-self.df['start_critical_subprocess1']).astype('timedelta64[m]')    
            
            self.df["start_process_reported_on_tower_diff"] = \
            (self.df['reported_on_tower']-self.df['start_process']).astype('timedelta64[m]')
            
            self.df["process_end_reported_on_tower_diff"] = \
            (self.df['reported_on_tower']-self.df['process_end']).astype('timedelta64[m]')
        
            print("done!")
            
        except Exception as e:
            print(e)
        
        return self.df

    def drop_datetime_cols(self)-> pd.DataFrame:
        
        '''
        this function drops datetime columns from a given dataframe

        Returns
        -------
        df : dataframe
            dataframe without cols of time datetime
        '''
                
        try:
            # get datetime features
            dt_cols = list(self.df.columns[self.df.dtypes=='datetime64[ns]'])

            # drop dt columns
            print("delete datetime features after encoding them... ")

            self.df= self.df.drop(dt_cols, axis = 1)
        
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df

    def impute_missing_data(self)-> pd.DataFrame:
        
        '''
        this fuunction fill in missing data into pandas dataframe
        using imputation criteria and list of grouping columns

        Returns
        -------
        df : pandas df
            dataframe without missing values/nans
        '''
       
        try:
            # impute missing data
            print("impute missing data... ")            

            for c in list(self.df.columns):  
                self.df[c] = self.df[c].astype(object)                
                self.df[c] = self.df[c].fillna(self.df.groupby(self.impgroupcol)[c].transform(self.impcriteria))
                            
            print("done!")
            
        except Exception as e:
            print(e)
           
        return self.df

    def drop_low_var_cols(self)-> pd.DataFrame:    
        
        '''
        this function drops columns with low variance/Std 
        i.e. std below given threshold

        Returns
        -------
        df : pandas df
            dataframe without low variance cols
        '''
                
        try:            
            # drop all na columns
            print("drop low-variance features... ")

            self.df= self.df.dropna(axis=1, how='all')    
            
            # keep features only with high variance threshold
            self.df = self.df.loc[:, self.df.std() > self.minstdthreshold]
            
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df

    def split_features_labels(self)-> Tuple[pd.DataFrame, pd.DataFrame]:    
        
        '''
        this function splits a dataset into features and labels subsets
                
        Returns
        -------
        features : pandas df
            dataframe of data features (independent variables) only
        labels: pandas df
            1-col dataframe of target variable
        '''        
        
        try:            
            # separate features and target into 2 variables
            print("split features & labels into 2 datasets... ")

            features = self.df
            features= features.drop(self.indexingcols, axis=1)
            labels = self.df[self.targetvar].copy()
        
            print("done!")
            
        except Exception as e:
            print(e)
            
        return features, labels

    def scale_data(self, features: pd.DataFrame)-> pd.DataFrame:
        
        '''
        this function scaled data features around 0 average (standardization)

        Parameters
        ----------
        features: pd.DataFrame
            features dataframe

        Returns
        -------
        pandas df
            dataframe of standardized columns
        '''
                
        try:            
            # create scaler instance            
            trans = StandardScaler()
            
            # fit scaler using numerical data
            scaled_features = trans.fit_transform(features.values)
            
            # convert the array back to a dataframe
            print("scale data around 0 average (standardization)... ")

            features= pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
            
            print("done!")
            
        except Exception as e:
            print(e)
            
        return features 
         
    def transform_data(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        
        '''
        this function applies all transformation methods and returns scaled features
        and target variable
               
        Returns
        -------
        pandas df
            dataframe of standardized columns
        pandas df
            target variable
        '''

        try:
            # create empty dataframes
            features = pd.DataFrame()
            target = pd.DataFrame()

            print("\nStart Data Transformation for: ", self.datasettype)

            # drop columns with high ratio of missing data
            self.df = self.drop_low_inf_cols()            

            # set datetime features to (datetime) data type
            self.df = self.get_datetime_features()         

            # get categorical features           
            cat_list = self.get_cat_features()            
            
            # drop heterogeneous cols from data
            self.df, cat_list = self.drop_heterogeneous_cols(cat_list)
            
            # encode categorical features into dummy ones
            self.df = self.encode_cat_features(cat_list)
            
            # encode datetime features to numerical ones
            self.df = self.encode_datetime_features()
            
            # generate new timediff features out of existing datetime features
            self.df = self.add_new_timediff_features()
            
            # delete datetime features after encoding them
            self.df = self.drop_datetime_cols()
            
            # impute missing data
            self.df = self.impute_missing_data()
            
            # drop low-variance features
            self.df = self.drop_low_var_cols()
            
            # split labels & features of data
            features, target = self.split_features_labels()                        
            
            # scale data around 0 average (standardization)
            features = self.scale_data(features)

            print("\nFinish Data Transformation for: ", self.datasettype, "\n")
            
        except Exception as e:
            print(e)
            
        return features, target
