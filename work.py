#!/usr/bin/env python

# CSC422 P06
# Fall 2018

import sys

import pandas as pd

from sklearn.preprocessing import normalize
# from sklearn.cross_validation import StratifiedKFold

# Number of rows to load from dataset to debug reduce time
DEBUG_NUMBER_ROWS = None

# Load dataset from file. Optionally load only a portion to
# facilitate debugging.
in_df = pd.read_csv('creditcard.csv', nrows=DEBUG_NUMBER_ROWS)
print('Dataset loaded. Fradulent credit card transactions.')
print('Credit: Université Libre de Bruxelles (http://mlg.ulb.ac.be/)')
print('Download: https://www.kaggle.com/mlg-ulb/creditcardfraud')

print()

# Report label statistics
vc = in_df['Class'].value_counts()
print(('Number of records: {}\n' +
	'Positive Labels {}\n' +
	'Negative Labels {}\n')
	.format(in_df.shape[0], vc[1], vc[0]))

# Split and extract numpy arrays for records X and labels y
X = in_df.drop(['Class'], axis=1).values
y = in_df['Class'].values

def normalize_df_column(column_series):
	cmin = column_series.min()
	cmax = column_series.max()
	column_series.apply(lambda x: (x-cmin)/(cmax-cmin))


# perform under/over sampling

# run stratified kfold
# cv = StratifiedKFold()

# train each type of model and test them