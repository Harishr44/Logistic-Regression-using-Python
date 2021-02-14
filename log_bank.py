# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:57:34 2020

@author: Harish
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
bank= pd.read_csv("bank_data.csv")
bank.isnull().sum()

bank.y.value_counts()
# as there are no NA values in the data set so we need not fill the NA values

x=bank.iloc[:,0:31]
y=pd.DataFrame(bank.y)

x_new=x.iloc[:,0:6]
x_new.drop(["default"],axis=1,inplace=True)
x_new.drop(["age"],axis=1,inplace=True)

#Model building 
from scipy import stats
import statsmodels.formula.api as sm
model_log=sm.logit('y~x_new',data=bank).fit()

#at first i build the model with all the columns in x data set but i have found out insignificant p value for most of the variables
# then i eliminate all the variables which are not connected to deposition or not
# then i creat new dataframe X_new repeating same procedure
# so i have to eliminate futher two variables age and default as p value is insignificant


model_log.summary()
y_pred=model_log.predict(bank)

bank["pred_prob"]=y_pred
bank["Deposite_val"]=0
bank.loc[y_pred>=0.5,"Deposite_val"]=1
# if Deposite yes then 1, if not deposite then=0

from sklearn.metrics import classification_report
z=classification_report(bank.Deposite_val,bank.y)

conf_matrix=pd.crosstab(bank.y,bank.Deposite_val)
conf_matrix
acuracy=(39235+919)/(45211)
acuracy #88.81

from sklearn import metrics

fpr,tpr,threshold=metrics.roc_curve(bank.y,y_pred)

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
roc_auc=metrics.auc(fpr,tpr)
roc_auc
#0.8377~84%
# as it is high so our model is good model