import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')

#Detecting missing values
missing_val_train=pd.DataFrame(train_df.isnull().sum())
missing_val_test=pd.DataFrame(test_df.isnull().sum())
#we can see that there are no missing values in train and test data


#distribution of target value in train data
sns.countplot(train_df['target'])
#we can see that the data is not balancedwith respect to target variable

#outlier removal
features=train_df.iloc[:,2:]
 
for i in features:
    q75,q25 = np.percentile(features.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    train_df = train_df.drop(train_df[train_df.loc[:,i]<min].index)
    train_df = train_df.drop(train_df[train_df.loc[:,i]>max].index)
    
#lets check the distribution
#normal distribution of independent features
plt.figure(figsize=(40,200))
for i,j in enumerate(features):
    plt.subplot(50,4,i+1)
    plt.hist(train_df[j])
    plt.title(j)
#most of the features have normalised distribution

#Distribution of features with respect to target class
plt.figure(figsize=(40,200))
for i,j in enumerate(features):
    plt.subplot(50,4,i+1)
    sns.distplot(train_df[train_df['target']==0][j],hist=False,label='0',color='green')
    sns.distplot(train_df[train_df['target']==1][j],hist=False,label='1',color='red')
'''We can observe that there is a considerable number of features 
with significant different distribution for the two target values.
For example, var_0, var_1, var_2, var_5, var_9, var_13, var_106, var_109, var_139 and many others.
Also some features, like var_2, var_13, var_26, var_55, var_175, var_184, var_196 shows a 
distribution that resambles to a bivariate distribution'''

#distribution of features with respect to train and test data
plt.figure(figsize=(40,200))
for i,j in enumerate(features):
    plt.subplot(50,4,i+1)
    sns.distplot(train_df[j],hist=False,label='train',color='green')
    sns.distplot(test_df[j],hist=False,label='test',color='red')
#The train and test are well ballanced with respect to distribution of the numeric variables.        
    
#let's check for correlation
corr_df=train_df.corr()
sns.heatmap(corr_df)
#none of the features have correlation between them

#split into train and test
X = train_df.iloc[:,2:]
y = train_df[["target"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
#scale
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#as our data is imbalanced we need to re-sample it to avoid less accurate results
#oversapmpling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
x_train_over, y_train_over = ros.fit_resample(x_train, y_train)
x_test_over, y_test_over = ros.fit_resample(x_test, y_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=42)
lrc.fit(x_train_over, y_train_over)
  
y_pred_prob = lrc.predict_proba(x_test_over)

y_pred_pos = y_pred_prob[:, 1]

#AUC score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test_over, y_pred_pos)
print('AUC for LR: %.2f' % auc)

#plot ROC AUC
fpr, tpr, thresholds = roc_curve(y_test_over, y_pred_pos)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#to calculate precision and recall we need to convert probability into target class
y_pred_class = lrc.predict(x_test_over)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_over, y_pred_class)
print(cm)
TN,FP,FN,TP = confusion_matrix(y_test_over, y_pred_class).ravel()

#precision and recall
lr_precision = TP*100/float(TP + FP)
lr_recall = TP*100/float(TP+FN)
print("precision and recall for LR: ",lr_precision, lr_recall)


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10, random_state=0, n_estimators=100)
rfc.fit(x_train_over, y_train_over)

y_rfc_prob = rfc.predict_proba(x_test_over)

#feature importance with random forest
imp_col=train_df.columns[2:]
importances = rfc.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10,50))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), imp_col[indices])
plt.xlabel('Relative Importance')

y_rfc_pos = y_rfc_prob[:, 1]

#AUC score
auc = roc_auc_score(y_test_over, y_rfc_pos)
print('AUC for RF: %.2f' % auc)

#AUC ROC curve
fpr, tpr, thresholds = roc_curve(y_test_over, y_rfc_pos)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#to calculate precision and recall we need to convert probability into target class
y_rfc_class = rfc.predict(x_test_over)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_over, y_rfc_class)
print(cm)
TN,FP,FN,TP = confusion_matrix(y_test_over, y_rfc_class).ravel()

#precision and recall
lr_precision = TP*100/float(TP + FP)
lr_recall = TP*100/float(TP+FN)
print("precision and recall for RF: ",lr_precision, lr_recall)


#naive bayes
from sklearn.naive_bayes import GaussianNB    
nb = GaussianNB()
nb.fit(x_train_over, y_train_over)

y_nb_prob = nb.predict_proba(x_test_over)

y_nb_pos = y_nb_prob[:, 1]

#AUC score
auc = roc_auc_score(y_test_over, y_nb_pos)
print('AUC for NB: %.2f' % auc)

#AUC ROC curve
fpr, tpr, thresholds = roc_curve(y_test_over, y_nb_pos)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#to calculate precision and recall we need to convert probability into target class
y_nb_class = nb.predict(x_test_over)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_over, y_nb_class)
print(cm)
TN,FP,FN,TP = confusion_matrix(y_test_over, y_nb_class).ravel()

#precision and recall
nb_precision = TP*100/float(TP + FP)
nb_recall = TP*100/float(TP+FN)
print("precision and recall for NB: ",nb_precision, nb_recall)

#removing outliers from test data
features_test=test_df.iloc[:,1:]
 
for i in features_test:
    q75,q25 = np.percentile(features_test.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    
    test_df = test_df.drop(test_df[test_df.loc[:,i]<min].index)
    test_df = test_df.drop(test_df[test_df.loc[:,i]>max].index)


#scale
test_scale = sc.transform(test_df.drop(['ID_code'],axis=1))

#impute naive bayes model into the test data as it has the highest AUC,precision,recall score
test_model=nb.predict(test_scale)
test_df["predicted_class"]=test_model

test_df.to_csv("predicted_python.csv", index=False)




















