===================================
Welcome to the DS Challenge Project
===================================


=================
Project Structure
=================

The project is built upon the following modules:

===================================
DS_CHALLENGE_Data_Access.DataAccess
===================================

This module is responsible of importing data from I/O devices In current version this module import data from .CSV files

=============================================
DS_CHALLENGE_Data_Preparation.DataPreparation
=============================================

This module is responsible of data preparation procedures. These procedures includes the following tasks:

1- add names to unnamed columns in the raw dataset
2- merge raw and target datasets into one dataframe
3- split merged dataframe into train/test datasets

===================================================
DS_CHALLENGE_Data_Transformation.DataTransformation
===================================================

This module applies different transformation procedures on the row dataset to be ready for the learning process

=======================================
DS_CHALLENGE_Data_Learning.DataLearning
=======================================

This module builds ML models on the dataset and generate predictions out of them

===============
Run the Project
===============

to execute the project type the following line from the root directory:

python main.py

=======================
View Evaluation Results
=======================

Evaluation results can be found under /fig/ directory

==============
Data Directory
==============

Data .CSV files are saved in /data/ directory

========================
Change pipeline settings
========================

To change the settings of the above modules, please consider changing the parameters in the 
JSON files on root directory. The JSON files are classified by name as follows:

1- datafiles_settings.JSON for DS_CHALLENGE_Data_Access.DataAccess 
2- data_prep_settings.JSON for DS_CHALLENGE_Data_Preparation.DataPreparation
3- data_trans_settings.JSON for DS_CHALLENGE_Data_Transformation.DataTransformation 










