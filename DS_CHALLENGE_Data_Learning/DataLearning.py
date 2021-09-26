# -*- coding: utf-8 -*-

'''
This class applies the following ML algorithms on dataset and evaluate 
them using RMSE score

1- Linear Regression
2- Decision Tree
'''

__author__ = "Boutros El-Gamil"
__copyright__ = "Copyright 2021"
__version__ = "0.1"
__maintainer__ = "Boutros El-Gamil"
__email__ = "contact@data-automaton.com"
__status__ = "Development"  

import operator
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

class DataLearning:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    train_target: pd.DataFrame, test_target: pd.DataFrame):
        
        # train dataset
        self.train_df = train_df
        
        # test dataset
        self.test_df = test_df
        
        # train target
        self.train_target = train_target
        
        # test target
        self.test_target = test_target    

    # init property of train df
    train_df = property(operator.attrgetter('_train_df'))

    # apply a non-empty check property on train dataset
    @train_df.setter
    def train_df(self, df1):
        if df1.empty: 
            raise Exception("train_df cannot be empty")
        self._train_df = df1
    
    def commonize_train_test_features(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        
        '''
        this function commonize set of features bw train and test datasets 
        (to fit properly in the training algorithm )

        Returns
        -------
        self.train_df : pandas df
            train dataset with same cols as test set
        self.test_df : pandas df
            test dataset with same cols as train set
        '''
                
        try:
            print("commonize train/test features... ")

            # get list of common columns between train & test datasets            
            common_cols= list(self.train_df.columns.intersection(self.test_df.columns))
            
            # reset train & test sets on shared columns (for training purpose)
            self.train_df = self.train_df[common_cols]
            self.test_df = self.test_df[common_cols]
            
            print("done!")
            
        except Exception as e:
            print(e)
            
        return self.train_df, self.test_df   
    
    def modeling_LR(self)-> LinearRegression:
        
        '''
        this function builds a linear regression model on the dataset

        Returns
        -------
        lin_model : LinearRegression
            linear regression model trained on train df
        '''

        try:   
            print("model data using Linear Regression regressor... ")

            # set LR regression instance
            lin_model = LinearRegression()

            # fit model to train data
            lin_model.fit(self.train_df, self.train_target)            

            print("done!")
            
        except Exception as e:
            print(e)

        return lin_model

    def modeling_DT(self)-> DecisionTreeRegressor:
        
        '''
        this function builds a decision tree regressor model on the dataset

        Returns
        -------
        tree_model : DecisionTreeRegressor
            decision tree regressor model trained on train df
        '''
        try:   
            print("model data using Decision Tree regressor... ")

            # set DT regression instance
            tree_model = DecisionTreeRegressor()

            # fit model to train data
            tree_model.fit(self.train_df, self.train_target)            

            print("done!")
            
        except Exception as e:
            print(e)

        return tree_model

    def evaluate_regressor(self, regname: str, model) -> Tuple[list, float]:

        '''
        this function evaluates a given regression models using 
        rmsr scores on both 10-fold cross validations on train df
        and test df

        Parameters
        ----------
        regname : str
            regressor name
        model: [regresson model]
            regressor model instance

        Returns
        -------
        lin_scores : list
            list of 10-fold CV RMSE scores on train data
        lin_rmse: flot
            RMSE score on test data
        '''

        try: 
            print("evaluate", regname, "model...")
            
            # get RMSE stats over 10-fold cross validation 
            lin_scores = cross_val_score(model, self.train_df, self.train_target,
                            scoring="neg_mean_squared_error", cv=10)
        
            # get predictions on test data
            predictions = model.predict(self.test_df)

            # get RMSE on test data
            lin_mse = mean_squared_error(self.test_target, predictions)
            lin_rmse = np.sqrt(lin_mse)

            print("done!")
            
        except Exception as e:
            print(e)

        return lin_scores, lin_rmse

    def print_evaluation_stats(self, regname: str, scores: list, rmse_test: float):

        '''
        this function prints evaluation results of a regressor

        Parameters
        ----------
        regname : str
            regressor name
        scores: list
            list of 10-fold CV RMSE scores on train data
        rmse_test: float
            RMSE score on test data        
        '''

        try:
            # print RMSE results of a regressor
            print("\n", regname, " Model\n 10-fold Cross Validation RMSE:")
            print("RMSE Mean: ", round(np.absolute(scores.mean()),3))
            print("RMSE Std: ", round(np.absolute(scores.std()),3))
            print("\nRMSE on Test data: ", round(rmse_test,3))                

        except Exception as e:
            print(e)

    def plot_rmse(self, regname: str, scores: list, rmse_test: float):
        
        '''
        this function plots evaluation results of a regressor

        Parameters
        ----------
        regname : str
            regressor name
        scores: list
            list of 10-fold CV RMSE scores on train data
        rmse_test: float
            RMSE score on test data        
        '''

        # set x,y axes
        x= list(range(1,11))
        y= np.log(np.absolute(scores[0:10]))

        # Plot log RMSE CV data
        plt.plot(x,y, label='log RMSE', marker='o')

        # Plot the average log RMSE line
        plt.plot(x, [np.mean(y)]*len(x), label='Mean RMSE', linestyle='--')

        # Make a legend
        plt.legend(loc="upper left") 

        # set titles fonts
        font1 = {'family':'serif','color':'blue','size':14}
        font2 = {'family':'serif','color':'blue','size':12}

        # set titles
        plt.title(regname + " Evaluation \n RMSE on Test Data= " + \
        str(round(rmse_test,3)), fontdict = font1)

        plt.xlabel("10-fold Cross Validation Runs", fontdict = font2)
        plt.ylabel("Log RMSE", fontdict = font2)
        plt.grid()

        # save plot to fig/ directory
        plt.savefig('fig/' + regname + '_eval',  dpi=600)  

        plt.show()

        # clear plot
        plt.clf()

        #plt.show() 
            
    def learn_data(self):
        '''
        this function applies all transformation methods and returns scaled features
        and target variable                       
        '''

        try:
            print("\nStart Data Learning... ")

            # commonize train & test features
            self.train_df, self.test_df = self.commonize_train_test_features()
            
            # Linear Regression modeling
            lr_model = self.modeling_LR()
            lr_model_scores, lr_rmse = self.evaluate_regressor("Linear Regression", lr_model)

            # Decision Tree modeling
            dt_model = self.modeling_DT()
            dt_model_scores, dt_rmse = self.evaluate_regressor("Decision Tree", dt_model)

            # print evaluation results
            self.print_evaluation_stats("Linear Regression", lr_model_scores, lr_rmse)
            self.print_evaluation_stats("Decision Tree", dt_model_scores, dt_rmse)

            # plot evaluation results
            self.plot_rmse("Linear Regression", lr_model_scores, lr_rmse)
            self.plot_rmse("Decision Tree", dt_model_scores, dt_rmse)

            print("\nFinish Data Learning ", "\n")
            
        except Exception as e:
            print(e)
            
        
