# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:15:59 2020

@author: Saeesha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("E:\\PROJECT EXCELR\\bank_final.csv")

data.head()

data.dropna(subset=['Name','City','State','Bank','BankState','DisbursementDate','MIS_Status'],inplace=True)
data.dropna(subset=['RevLineCr'],inplace=True)
data.drop_duplicates()

dups = data[data.duplicated(keep='first')].index
data.drop(dups, inplace=True)

#drop chgoffdate (no.of null values)and name,city as  UNIQUE values present
data.drop(['ChgOffDate','Name','City'],axis=1,inplace=True)

#conversion of data types

data['DisbursementDate'] = pd.to_datetime(data['DisbursementDate'])
data['ApprovalDate'] = pd.to_datetime(data['ApprovalDate'])



def clean_currency(x):
   
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)
    
data['DisbursementGross']= data['DisbursementGross'].apply(clean_currency).astype('float')

data['ChgOffPrinGr']=data['ChgOffPrinGr'].apply(clean_currency).astype('float')

data['GrAppv']= data['GrAppv'].apply(clean_currency).astype('float')
data['BalanceGross']=data['BalanceGross'].apply(clean_currency).astype('float')
data['SBA_Appv']= data['SBA_Appv'].apply(clean_currency).astype('float')



# Dropping rows with garbage values    
data.drop(data[data['LowDoc'] == '1'].index, inplace = True) # 1
data.drop(data[data['LowDoc'] == 'C'].index, inplace = True) # 83
data.drop(data[data['RevLineCr'] == ','].index, inplace = True) # 1
data.drop(data[data['RevLineCr'] == '1'].index, inplace = True) # 3
data.drop(data[data['RevLineCr'] == '`'].index, inplace = True) # 2
data.drop(data[data['RevLineCr'] == 'T'].index, inplace = True) # 4810
data.drop(data[data['RevLineCr'] == '0'].index, inplace = True)

#Dropping the garbage values
data.drop(data[data['NewExist'] == 0].index, inplace = True) # 124
data.drop(data[data['FranchiseCode'] > 1].index, inplace = True)


cleanup_nums = {"LowDoc":     {"N": 0, "Y": 1}}
data.replace(cleanup_nums, inplace=True)
data["LowDoc"].value_counts()

#data['DaysforDibursement']


var_revlinecr=data['RevLineCr']
var_rev=[]
for i in var_revlinecr:
    i=str(i).strip()
    if i == "N":
        var_rev.append(0)
    else:
        var_rev.append(1)
data["RevLineCr"]= var_rev
data["RevLineCr"].value_counts()



var_mis_status=data['MIS_Status']
var_mis=[]

for j in var_mis_status:
    j=str(j).strip()
    if j == "CHGOFF":
        var_mis.append(0)
    else:
        var_mis.append(1)
    
        

data['MIS_Status']=var_mis
data['MIS_Status'].value_counts()

data['DaysforDibursement'] = data['DisbursementDate'] - data['ApprovalDate']
data['DaysforDibursement'] = data.apply(lambda row: row.DaysforDibursement.days, axis=1)
#Removing the Date-time variables ApprovalDate and DisbursementDate
data=data.drop(['ApprovalDate','DisbursementDate'],axis=1)


data['Bank']=data['Bank'].astype('category')
data['State']=data['Bank'].astype('category')
data['BankState']=data['BankState'].astype('category')
data['Bank']=data['Bank'].cat.codes
data['State']=data['State'].cat.codes
data['BankState']=data['BankState'].cat.codes


#correlations
data_corr=data.corr()
data_corr


mask = np.triu(np.ones_like(data_corr, dtype=np.bool))
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 14))
    ax = sns.heatmap(data_corr, mask=mask,vmin=-1, annot=True,vmax=1,cmap='Blues' ,square=True)

data.drop(['ApprovalFY','GrAppv','DisbursementGross','Bank','BankState'],axis=1,inplace=True)


#correlations
data_corr=data.corr()
data_corr


mask = np.triu(np.ones_like(data_corr, dtype=np.bool))
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 14))
    ax = sns.heatmap(data_corr, mask=mask,vmin=-1, annot=True,vmax=1,cmap='Blues' ,square=True)
#data.columns

data.drop(['CCSC','BalanceGross','State','Zip','RevLineCr'],axis=1,inplace=True)


columns=data.columns.tolist()
columns=[c for c in columns if c not in ['MIS_Status']]
tar='MIS_Status'



state=np.random.RandomState(42)
x=data[columns]
y=data[tar]
print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

from imblearn.over_sampling import SMOTE
smote=SMOTE( random_state=42)

x_train_smote,y_train_smote=smote.fit_sample(x_train,y_train)

from collections import Counter
print("before SMOTE:",Counter(y_train))
print('after SMOTE:',Counter(y_train_smote))


print('Distribution of the Classes in the subsample dataset')
sns.countplot(y_train_smote, data=data)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


from xgboost import XGBClassifier
# Linear SVM powered by SGD Classifier (params are defaults)
xgb = XGBClassifier(random_state=29,learning_rate=0.7)
xgb.fit(x_train_smote,y_train_smote)
y_pred=xgb.predict(x_test)
y_pred_tr=xgb.predict(x_train_smote)
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train_smote == y_pred_tr)/len(y_train_smote))
from sklearn.metrics import classification_report
print("Classification Report(Train)")
print(classification_report(y_train_smote, y_pred_tr))
print("Classification Report(Test)")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
cm
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, fmt='d', # fmt='d' gets rid of numbers like 1.8e + 02
xticklabels=['positive', 'negative','neutral'],
yticklabels=['positive', 'negative','neutral'])
plt.xlabel('True label', fontsize= 15)
plt.ylabel('Predicted label',fontsize= 15)

data.columns

import pickle
#saving model to disk
pickle.dump(xgb,open('modelxg.pkl','wb'))

# In[ ]:
#loading model to compare results
model=pickle.load(open('modelxg.pkl','rb'))










