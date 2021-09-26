# -*- coding: utf-8 -*-

'''
This class prepare dataset for transformation actions by applying the following tasks

    1- add names to unnamed columns in the raw dataset
    2- merge raw and target datasets into one dataframe
    3- split merged dataframe into train/test datasets
'''

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development"  

import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

class DataPreparation:
    def __init__(self, df: pd.DataFrame, target: pd.DataFrame):
        with open("./data_prep_settings.json") as dataprep_settings:

            # read module settings file
            dp_settings = json.load(dataprep_settings)  
            
            # list of un-named columns indeces and corresponded new names
            self.__unnamedcols = dp_settings["unnamed_cols"]  
            
            # list of common columns between raw & target dfs
            self.__commoncol = dp_settings["common_col"]          
            
            # groupby column to be considered during sampling
            self.__splitcol = dp_settings["split_col"]
            
            # number of shuffling iterations
            self.__nsplits= dp_settings["n_splits"]
            
            # integer for reproducible output over multiple calls
            self.__randstand= dp_settings["rand_stand"]            
            
            # size of test dataset
            self.__testsize = dp_settings["test_size"]        

        # raw dataset
        self.df = df

        # target dataset
        self.target= target

    def name_unnamed_cols(self)-> pd.DataFrame:
        '''
        this function add names to unnamed cols in a dataframe

        Returns
        -------
        df: pandas df
            dataframe with added col names
        '''
        
        try:                        
            # fill-in columns names
            print("name unnamed columns in raw df... ")

            for i in range(0,len(self.__unnamedcols)):                  
                self.df.columns.values[int(self.__unnamedcols[i]["id"])] = self.__unnamedcols[i]["name"]
                
            print("done!")
        
        except Exception as e:
            print(e)    

        return self.df

    def merge_data(self)-> pd.DataFrame:
        '''
        this function merges 2 dataframes into one using list of common columns        

        Returns
        -------
        df: pandas df
            merged dataframe of input dataframs.
        '''                
        
        try:     
            # merge raw & target datasets
            print("merge raw & target dfs... ")
            self.df = pd.merge(self.df, self.target, on=(self.__commoncol))
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.df

    def split_data(self)-> Tuple[pd.DataFrame, pd.DataFrame]:    
        '''
        this function splits dataframe into train/ test datasets

        Returns
        -------
        train : pandas df
            dataframe of train data
        test : pandas df
            dataframe of train data
        '''
                
        try:
            # init StratifiedShuffleSplit instance
            split = StratifiedShuffleSplit(n_splits=self.__nsplits, 
            test_size=self.__testsize, random_state=self.__randstand)

            # get train/test datasets out of df
            print("split data into train/test datasets... ")

            for train_index, test_index in split.split(self.df, self.df[self.__splitcol]):
                train = self.df.loc[train_index]
                test = self.df.loc[test_index]
            
            print("done!")
            
        except Exception as e:
            print(e)
            
        return train, test

    def prepare_data(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        this function returns prepared train/ test datasets

        Returns
        -------
        train : pandas df
            dataframe of train data
        test : pandas df
            dataframe of train data
        '''

        try:
            print("Start Data Preparation... ")

            # 1- add names to unnamed columns in raw dataset                
            self.df = self.name_unnamed_cols()
            
            # 2- merge raw and target dataset into one dataframe
            self.df = self.merge_data()
            
            # 3- split dataframe into train/test datasets
            train, test = self.split_data()

            print("Finish Data Preparation!\n")

        except Exception as e:
            print(e)

        return train, test
