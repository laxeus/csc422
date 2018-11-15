#!/usr/bin/env python

# CSC422 P06
# Fall 2018

import sys

import pandas as pd
import numpy as np
from scipy import interp

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold

from sklearn.utils.class_weight import compute_class_weight

# Number of rows to load from dataset to debug reduce time
DEBUG_NUMBER_ROWS = 20000

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

print('Class weights:', compute_class_weight('balanced', [0,1], y))

# def normalize_df_column(column_series):
# 	cmin = column_series.min()
# 	cmax = column_series.max()
# 	column_series = column_series.apply(lambda x: (x-cmin)/(cmax-cmin))



# perform under/over sampling

# run stratified kfold
cv = StratifiedKFold(n_splits=5)
# TODO: change the class weights!
clf = SVC(gamma='auto', kernel='rbf', probability=True,class_weight={1: 100})

# ROC curve variables
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0

# TODO: remove this line! take only some features to train faster
# TODO: use feature_selection to choose the best features instead of doing this!
X = X[:,1:5]

# perform cross validation
## train each type of model and test them

# TODO: this is incorrect! replace this form of cross validation with one
# which accounts for imbalanced classes! import the imbalance learn package!
for train, test in cv.split(X,y):
	probas_ = clf.fit(X[train],y[train]).predict_proba(X[test])
	# Compute ROC curve and area the curve
	fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	tprs.append(interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.plot(fpr, tpr, lw=1, alpha=0.3,
		label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

	print("Split {} done!".format(i))
	i += 1



# plot the cross validation ROC curve
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


