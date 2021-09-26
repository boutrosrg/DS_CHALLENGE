# -*- coding: utf-8 -*-

'''
This class implements data files access methods
'''

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development"  

import os
import json
import pandas as pd

class DataAccess:

    def __init__(self):
        
        with open("./datafiles_settings.json") as datafiles_settings:

            # read module settings file
            df_settings = json.load(datafiles_settings)            
            
            # import data path
            self.__path = df_settings["path"]
            
            # import raw file name
            self.__rawfilename= df_settings["raw_data_file"]
            
            # import target file name
            self.__targetfilename= df_settings["target_data_file"]            
            
            # import CSV header settings
            self.__csvheader = df_settings["csv_header"]
            
            # import CSV separator
            self.__separator = df_settings["separator"]        

    def import_csv_file(self, filetype: str)-> pd.DataFrame:

        '''
        this function imports data from .csv file into pandas dataframe

        Parameters
        ----------
        filetype : string
            type of datafile        

        Returns
        -------
        df: pandas df
            dataframe of imported file
        '''

        try:
            # set data file types 
            datafile_types = ['rawdata', 'target']

            # check data file type validation
            if filetype not in datafile_types:
                raise ValueError("Invalid file type. Expected one of: %s" % datafile_types)

            print("Import Datafile: ", filetype)

            # set file path
            file_path= os.path.join(self.__path, self.__rawfilename) \
                        if filetype== 'rawdata' \
                        else os.path.join(self.__path, self.__targetfilename)

            # read file into pandas df
            df= pd.read_csv(file_path, header=self.__csvheader, sep=self.__separator)
            
            print("Done!\n")

        except Exception as e:
            print(e) 

        return df



