import os 
import json
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


def build_and_train():

	data = pd.read_csv('../data/training.csv')
	data = data.dropna(subset=['Gender', 'Married', 'Credit_History', 'LoanAmount'])

	pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome',\
				'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

	X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['Loan_Status'], \
														test_size=0.25, random_state=42)
	y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()
	y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()

	pipe = make_pipeline(PreProcessing(),
						RandomForestClassifier())

	param_grid = {"randomforestclassifier__n_estimators" : [10, 20, 30],
				 "randomforestclassifier__max_depth" : [None, 6, 8, 10],
				 "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 20], 
				 "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}

	grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)

	grid.fit(X_train, y_train)

	return(grid)


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """
        pred_var = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',\
                    'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
        
        df = df[pred_var]
        
        df['Dependents'] = df['Dependents'].fillna(0)
        df['Self_Employed'] = df['Self_Employed'].fillna('No')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(self.term_mean_)
        df['Credit_History'] = df['Credit_History'].fillna(1)
        df['Married'] = df['Married'].fillna('No')
        df['Gender'] = df['Gender'].fillna('Male')
        df['LoanAmount'] = df['LoanAmount'].fillna(self.amt_mean_)
        
        gender_values = {'Female' : 0, 'Male' : 1} 
        married_values = {'No' : 0, 'Yes' : 1}
        education_values = {'Graduate' : 0, 'Not Graduate' : 1}
        employed_values = {'No' : 0, 'Yes' : 1}
        property_values = {'Rural' : 0, 'Urban' : 1, 'Semiurban' : 2}
        dependent_values = {'3+': 3, '0': 0, '2': 2, '1': 1}
        df.replace({'Gender': gender_values, 'Married': married_values, 'Education': education_values, \
                    'Self_Employed': employed_values, 'Property_Area': property_values, \
                    'Dependents': dependent_values}, inplace=True)
        
        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
        """
        
        self.term_mean_ = df['Loan_Amount_Term'].mean()
        self.amt_mean_ = df['LoanAmount'].mean()
        return self

if __name__ == '__main__':
	model = build_and_train()

	filename = 'model_v1.pk'
	with open('../flask_api/models/'+filename, 'wb') as file:
		pickle.dump(model, file)