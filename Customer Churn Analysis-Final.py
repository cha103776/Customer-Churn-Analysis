#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import re
#import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')


# In[7]:


df = pd.read_csv("Customer_churn.csv")
df.head(5)


# In[8]:


df.columns


# In[ ]:


#Since there are no null values then we can proceed#


# # checking the datatype

# In[12]:


df.dtypes


# In[13]:


df['TotalCharges'].unique()


# In[14]:


df.loc[df['TotalCharges']==" "]


# In[15]:


df['TotalCharges']=df['TotalCharges'].replace(" ",np.nan)


# In[17]:


df.isnull().sum()


# In[18]:


#Converting the column type from object to flot 


# In[19]:


df['TotalCharges']=df['TotalCharges'].astype(float)


# In[20]:


#now we can see total charges coloumn has some spaces values which is replaced by nan values
#handle the nan values and replacing it with mean of coloumn . as the coloumn is continuous in nature


# In[21]:


df['TotalCharges']=df['TotalCharges'].fillna(np.mean(df['TotalCharges']))


# In[22]:


df.isnull().sum()


# In[23]:


df.dtypes


# In[24]:


#Making the DataFrame for nominal data


# In[25]:


df_visualization_nominal=df[['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']].copy()


# In[26]:


df_visualization_nominal.columns


# In[27]:


#Visualization of data


# In[28]:


ax=sns.countplot(x='gender',data=df_visualization_nominal)
print(df_visualization_nominal['gender'].value_counts())


# In[29]:


ax=sns.countplot(x='SeniorCitizen',data=df_visualization_nominal)
print(df_visualization_nominal['SeniorCitizen'].value_counts())


# In[30]:


ax=sns.countplot(x='Partner',data=df_visualization_nominal)
print(df_visualization_nominal['Partner'].value_counts())


# In[31]:


ax=sns.countplot(x='Dependents',data=df_visualization_nominal)
print(df_visualization_nominal['Dependents'].value_counts())


# In[32]:


ax=sns.countplot(x='PhoneService',data=df_visualization_nominal)
print(df_visualization_nominal['PhoneService'].value_counts())


# In[33]:


ax=sns.countplot(x='MultipleLines',data=df_visualization_nominal)
print(df_visualization_nominal['MultipleLines'].value_counts())


# In[34]:


ax=sns.countplot(x='InternetService',data=df_visualization_nominal)
print(df_visualization_nominal['InternetService'].value_counts())


# In[35]:


ax=sns.countplot(x='OnlineSecurity',data=df_visualization_nominal)
print(df_visualization_nominal['OnlineSecurity'].value_counts())


# In[36]:


ax=sns.countplot(x='OnlineBackup',data=df_visualization_nominal)
print(df_visualization_nominal['OnlineBackup'].value_counts())


# In[37]:


ax=sns.countplot(x='DeviceProtection',data=df_visualization_nominal)
print(df_visualization_nominal['DeviceProtection'].value_counts())


# In[38]:


ax=sns.countplot(x='TechSupport',data=df_visualization_nominal)
print(df_visualization_nominal['TechSupport'].value_counts())


# In[39]:


ax=sns.countplot(x='StreamingTV',data=df_visualization_nominal)
print(df_visualization_nominal['StreamingTV'].value_counts())


# In[40]:


ax=sns.countplot(x='StreamingMovies',data=df_visualization_nominal)
print(df_visualization_nominal['StreamingMovies'].value_counts())


# In[41]:


ax=sns.countplot(x='Contract',data=df_visualization_nominal)
print(df_visualization_nominal['Contract'].value_counts())


# In[42]:


ax=sns.countplot(x='PaperlessBilling',data=df_visualization_nominal)
print(df_visualization_nominal['PaperlessBilling'].value_counts())


# In[43]:


ax=sns.countplot(x='PaymentMethod',data=df_visualization_nominal)
print(df_visualization_nominal['PaymentMethod'].value_counts())


# In[44]:


ax=sns.countplot(x='Churn',data=df_visualization_nominal)
print(df_visualization_nominal['Churn'].value_counts())


# In[45]:


df_visualization_ordinal=df[['customerID','tenure']].copy()


# In[46]:


sns.catplot(x='SeniorCitizen',y='tenure',data=df)


# In[47]:


#checking the distribution of the continuous value of the float type columns


# In[48]:


df_visualization_continuous=df[['MonthlyCharges','TotalCharges']].copy()


# In[49]:


sns.distplot(df_visualization_continuous['TotalCharges'],kde=True)


# In[50]:


sns.distplot(df_visualization_continuous['MonthlyCharges'],kde=True)


# In[51]:


# Encoding of DataFrame


# In[52]:


from sklearn.preprocessing import OrdinalEncoder
enc= OrdinalEncoder()


# In[53]:


for i in df.columns:
    if df[i].dtype=='object':
        df[i]=enc.fit_transform(df[i].values.reshape(-1,1))


# In[54]:


df


# In[55]:


#describe the dataset


# In[56]:


df.describe()


# In[57]:


plt.figure(figsize=[22,7])
sns.heatmap(df.describe(),annot=True,linewidth=0.1,linecolor='black',fmt='0.2f')
plt.show()


# In[58]:


#We have to check the correlation with target columns


# In[59]:


df.corr()


# In[60]:


plt.figure(figsize=[22,7])
sns.heatmap(df.corr(),annot=True,linewidth=0.1,linecolor='black',fmt='0.2f')
plt.show()


# In[61]:


plt.figure(figsize=[22,7])
df.corr()['Churn'].sort_values(ascending=False).drop(['Churn']).plot(kind='bar',color='c')
plt.xlabel('Features',fontsize=14)
plt.ylabel('column with target name',fontsize=14)
plt.title('correlation',fontsize=18)
plt.show()


# In[62]:


df.skew()


# In[63]:


#Skewness should be below 0.5 , the columns are SeniorCitizen,Dependents,Phoneservices,contract, Total charges & churn does not having the range, we will not treat that


# In[64]:


df.dtypes


# In[65]:


# Checking and Removing outliers 


# In[66]:


df['TotalCharges'].plot.box()


# In[67]:


df['SeniorCitizen'].plot.box()


# In[68]:


df['MonthlyCharges'].plot.box()


# In[69]:


# considering Outlier removal


# In[70]:


df.shape


# In[71]:


from scipy.stats import zscore
import numpy as np
z=np.abs(zscore(df))
threshold=3
np.where(z>3)


# In[72]:


df_new_z=df[(z<3).all(axis=1)]
df_new_z


# In[73]:


df_new_z.shape


# In[74]:


# % of data loss after outlier removal


# In[75]:


Data_loss=((7043-6361)/7043*100)


# In[76]:


Data_loss


# In[77]:


#Our data is becoming biased not considering senior citizen


# In[78]:


# Separating the column into features (x) and target(y)


# In[79]:


X=df.drop('Churn',axis=1)
Y=df['Churn']


# In[80]:


#Scaling the data using min max scaler


# In[ ]:


#pip install -U scikit-learn scipy matplotlib


# In[81]:


from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[ ]:


# Machine Learning Model


# In[ ]:


### intitilze LinearRegression model

#for i in range(1,200):
   # X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=i)
   # lr=LinearRegression()
   # lr.fit(X_train,Y_train)
    #pred_train=lr.predict(X_train)
    #pred_test=lr.predict(X_test)
    #print(f"At random state {i},the training accuracy is :-{r2_score(Y_train,pred_train)}")
    #print(f"At random state {i},the testing accuracy is :-{r2_score(Y_test,pred_test)}")
    #print("\n")


# In[82]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=12)


# In[ ]:


#lr.fit(X_train,Y_train)


# In[ ]:


#pred_test=lr.predict(X_test)


# In[ ]:


#print(r2_score(Y_test,pred_test))


# In[ ]:


#Cross-Validation of model


# In[ ]:


#Train_accuracy=r2_score(Y_train,pred_train)
#Test_accuracy=r2_score(Y_test,pred_test)
#from sklearn.model_selection import cross_val_score
#for j in range (2,10):
    #cv_score=cross_val_score(lr,X,Y,cv=j)
   # cv_mean=cv_score.mean()
   # print(f"At cross fold {j}the cv score is {cv_mean} and accuracy score for training is {Train_accuracy} and accuracy for the testing is{Test_accuracy}")
    #print("\n")


# In[ ]:


#plt.figure(figsize=(8,6))
#plt.scatter(x=Y_test,y=pred_test, color="r")
#plt.plot(Y_test,Y_test, color="b")
#plt.xlabel('Actual churn',fontsize=14)
#plt.ylabel('Predicted churn',fontsize=14)
#plt.title('Linear Regression',fontsize=18)
#plt.show()


# # Logistic Regression

# In[83]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[84]:


### intitilze Logistic Regression model
maxAccu=0
maxRS=0
for i in range(1,200):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=i)
    lr=LogisticRegression()
    lr.fit(X_train,Y_train)
    pred_train=lr.predict(X_train)
    pred_test=lr.predict(X_test)
    acc=accuracy_score(Y_test,pred_test)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print('Best accuracy is',maxAccu,'on Random_state',maxRS)


# In[86]:


lr_pred= lr.predict(X_test)
report = classification_report(Y_test,lr_pred)


# In[87]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(Y_test,lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[88]:


Y_pred_prob = lr.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show();


# # Random Forest

# In[89]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [5,10,15],
    'n_estimators': [10,30,70,100]
}

rf=RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = 1, verbose = 2)

grid_search.fit(X_train,Y_train)


# In[90]:


print("best model parameter")
print(grid_search.best_estimator_)

### Modle Training Performance
print("model performance on train data\n")
Y_pred_train=grid_search.predict(X_train)
print('Confusion matrix for train data')

print(confusion_matrix(Y_train,Y_pred_train))

print('Classification Report for train data')
print(classification_report(Y_train,Y_pred_train))


train_accuracy=accuracy_score(Y_train,Y_pred_train)
print('accuracy score for train data:{:.2f}%'.format(train_accuracy*100))


print("\n\n")
print("model performance on test data\n")
Y_pred=grid_search.predict(X_test)
print('Confusion matrix for test data')

print(confusion_matrix(Y_test,Y_pred))

print('Classification Report for test data')
print(classification_report(Y_test,Y_pred))

from sklearn.metrics import accuracy_score
test_accuracy=accuracy_score(Y_test,Y_pred)
print('accuracy score for test data:{:.2f}%'.format(test_accuracy*100))


# # Support Vector classifire#

# In[91]:


from sklearn.svm import SVC  
 
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf']}


svc = SVC()
grid_search_svc = GridSearchCV(estimator = svc, param_grid = param_grid ,cv = 3, n_jobs = 1, verbose = 2)

grid_search_svc.fit(X_train,Y_train)


# In[92]:


print("best model parameter")
#print(grid_search_svc.best_estimator_)

### Modle Training Performance
print("model performance on train data\n")
Y_pred_train=grid_search_svc.predict(X_train)
print('Confusion matrix for train data')

print(confusion_matrix(Y_train,Y_pred_train))

print('Classification Report for train data')
print(classification_report(Y_train,Y_pred_train))

train_accuracy=accuracy_score(Y_train,Y_pred_train)
print('accuracy score for train data:{:.2f}%'.format(train_accuracy*100))


print("\n\n")
print("model performance on test data\n")
Y_pred=grid_search_svc.predict(X_test)
print('Confusion matrix for test data')

print(confusion_matrix(Y_test,Y_pred))

print('Classification Report for test data')
print(classification_report(Y_test,Y_pred))

from sklearn.metrics import accuracy_score
test_accuracy=accuracy_score(Y_test,Y_pred)
print('accuracy score for test data:{:.2f}%'.format(test_accuracy*100))


# # Decision Tree#

# In[93]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,Y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,Y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# In[94]:


print(classification_report(Y_test, predictdt_y))


# # AdaBoost Classifier

# In[103]:


a_model = AdaBoostClassifier()
a_model.fit(X_train,Y_train)
a_preds = a_model.predict(X_test)
print("AdaBoost Classifier accuracy")
metrics.accuracy_score(Y_test, a_preds)


# In[104]:


print(classification_report(Y_test, a_preds))


# In[105]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(Y_test, a_preds),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("AdaBoost Classifier Confusion Matrix",fontsize=14)
plt.show()


# # Gradient Boosting Classifier

# In[106]:


gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting Classifier", accuracy_score(Y_test, gb_pred))


# In[107]:


print(classification_report(Y_test, gb_pred))


# In[109]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(Y_test, gb_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("AdaBoost Classifier Confusion Matrix",fontsize=14)
plt.show()


# In[ ]:


### best model will save as joblib


### randomf forest classfier have good f1-score for both class (yes,no). so that we will use random forest classfier for futhure 
### prediction


### for SVC f1-score is very low. It's show that model is not perform well for test dataset.
### for class 0 (loan statu NO) has very low recall .


# In[110]:


# save the model to disk
filename = 'finalized_model.joblib'
joblib.dump(grid_search, filename)


# In[ ]:




