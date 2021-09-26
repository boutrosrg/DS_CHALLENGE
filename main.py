# -*- coding: utf-8 -*-

'''
This script runs the whole data pipeline
Please read the description of each class for more info about 
class job and methods 
'''

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development" 

from DS_CHALLENGE_Data_Access.DataAccess import DataAccess
from DS_CHALLENGE_Data_Preparation.DataPreparation import DataPreparation
from DS_CHALLENGE_Data_Transformation.DataTransformation import DataTransformation
from DS_CHALLENGE_Data_Learning.DataLearning import DataLearning

## 1- Get Data
# init DataAccess class
data_access= DataAccess()

# import raw & target datasets
raw= data_access.import_csv_file('rawdata')
target= data_access.import_csv_file('target')

## 2- Data Preparation
# init DataPreparation class
data_prep = DataPreparation(raw, target)

# apply preparation steps on dataset 
# and get corresponded train/test subsets
train_set, test_set= data_prep.prepare_data()

## 3- Data Transformation
# init DataTransformation class
data_trans1= DataTransformation(train_set, "train")
data_trans2= DataTransformation(test_set, "test")

# apply transformations on train/test datasets
train, train_labels= data_trans1.transform_data()
test, test_labels= data_trans2.transform_data()

## 4- ML & Evaluation
# init DataLearning class
data_learn= DataLearning(train, test, train_labels, test_labels)

# apply learning algorithms on train/test datasets
data_learn.learn_data()

