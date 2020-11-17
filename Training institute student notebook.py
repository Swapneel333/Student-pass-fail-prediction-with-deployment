#!/usr/bin/env python
# coding: utf-8

# # Training Insititute Student Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv(r'C:\Users\MY PC\Desktop\Train.csv')
test = pd.read_csv(r'C:\Users\MY PC\Desktop\Test.csv')
train.head()


# In[3]:


train.shape


# In[4]:


test.shape


# In[5]:


train.info()


# In[6]:


test.info()


# ## Dropping the Variables:
# - id
# - program_id
# - test_id
# - trainee_id
# 
# These variables would not have much impact on the model.

# In[7]:


train = train.drop(["id" , "program_id" , "test_id" , "trainee_id"] , axis = 1)
train.head()


# In[8]:


test = test.drop(["id" , "program_id" , "test_id" , "trainee_id"] , axis = 1)
test.head()


# In[44]:


test["is_handicapped"].value_counts()


# ## Missing Value Treatment

# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


train.age.hist(bins=50)
plt.show()


# In[12]:


test.age.hist(bins=50)
plt.show()


# ### Direct imputation for age column would not give true insights hence creating groups to impute for age of each category

# In[13]:


train.groupby(["program_type" ,"gender" , "education"])['age'].mean()


# In[14]:


train["age"] = train["age"].fillna(train.groupby(["program_type" ,"gender" , "education"])['age'].transform('mean'))


# In[15]:


test["age"] = test["age"].fillna(test.groupby(["program_type" , "gender" , "education"])['age'].transform('mean'))


# In[16]:


print(train["age"].isnull().sum())
print(test["age"].isnull().sum())


# In[17]:


sns.distplot(train.trainee_engagement_rating)
plt.show()


# In[18]:


sns.distplot(test.trainee_engagement_rating)
plt.show()


# In[19]:


### Filling missing values with mean for trainee_engagement_rating

train["trainee_engagement_rating"] = train["trainee_engagement_rating"].fillna(train["trainee_engagement_rating"].mean())
train["trainee_engagement_rating"].isnull().sum()


# In[20]:


### Filling missing values with mean for trainee_engagement_rating

test["trainee_engagement_rating"] = test["trainee_engagement_rating"].fillna(test["trainee_engagement_rating"].mean())
test["trainee_engagement_rating"].isnull().sum()


# In[21]:


x = train.drop(["is_pass"],axis=1)
y = train["is_pass"]


# # EDA

# In[22]:


num_columns = list(train._get_numeric_data().columns)
all_columns = list(train.columns)


# In[23]:


num_columns


# In[24]:


cat_columns = []
for i in all_columns:
    if i not in num_columns:
        cat_columns.append(i)
cat_columns


# ### CATAGORICAL FEATURES

# In[25]:


for i in cat_columns:
    plt.figure(figsize=(8,5))
    sns.countplot(train["is_pass"],hue=train[i])
    plt.legend(loc=2)
    plt.title(i)
    plt.tight_layout()
    plt.show()


# ### Analyzed Results
# 
# 1. Y-program type generates most pass outcomes and T-program type generates most failed outcomes.
# 2. In case of offline test type,pass ratio is more when compared to online test type.
# 3. Pass ratio is more when difficulty level is 'easy'.
# 4. More male trainees have passed in comparison with females.
# 5. High-school diploma trainees have passed in more number.
# 6. Non-handicapped trainees have passed in more number.

# ### NUMERICAL FEATURES

# In[26]:


plt.figure(figsize=(8,6))
sns.countplot(train['is_pass'],hue=train['city_tier'])
plt.legend(loc=2)
plt.title("City Tier")
plt.show()


# In[27]:


plt.figure(figsize=(8,6))
sns.countplot(train['program_duration'],hue=train['is_pass'])
plt.legend(loc=2)
plt.title("Passed/Failed as per Program Duration")
plt.show()


# In[28]:


plt.figure(figsize=(8,6))
sns.barplot(train["is_pass"],train["age"])
plt.show()


# In[29]:


plt.figure(figsize=(8,6))
sns.countplot(train['total_programs_enrolled'],hue=train['is_pass'])
plt.legend(loc=1)
plt.title("Passed/Failed as per total programs enrolled")
plt.show()


# In[30]:


plt.figure(figsize=(8,6))
sns.barplot(train["is_pass"],train["trainee_engagement_rating"])
plt.show()


# ### Analyzed Results
# 
# 1. For a 134 days Program duration followed by a 120 days program duration pass ration is high.
# 2. More trainees from tier-3 city have passed.
# 3. Trainees with 35+ age have passed in more number.
# 4. Trainees who have enrolled in 2 programs have passed in more number followed by trainees with 4-programs enrollment.
# 5. An average trainee engagement rating of more than 2 have resulted in more number of passed trainees. 

# # DATA PRE-PROCESSING

# In[31]:


train_cat = train[cat_columns]
train_num = train[num_columns]


# In[32]:


train_cat.shape


# In[33]:


train_cat_dummy = pd.get_dummies(train_cat,prefix_sep="_",drop_first=True)
train_cat_dummy.head()


# In[34]:


Train = train_num.join(train_cat_dummy)
Train.head()


# ### Pre-processing the Test dataset

# In[35]:


test_num = list(test._get_numeric_data().columns)
test_all = list(test.columns)
test_cat = []
for i in test_all:
    if i not in test_num:
        test_cat.append(i)
test_cat


# In[36]:


testnum = test[test_num]
testcat = test[test_cat]
testcat_dummy = pd.get_dummies(testcat,prefix_sep='_',drop_first=True)
testcat_dummy.head()


# In[37]:


Test = testnum.join(testcat_dummy)
Test.head()


# # MULTICOLLINEARITY CHECK

# In[38]:


plt.figure(figsize=(20,10))
sns.heatmap(abs(Train.corr()),annot=True,linewidths=0.3,cmap="Set2")
plt.show()


# In[39]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
X_data = Train.drop(["is_pass"],axis=1)
Y_data = Train["is_pass"]


# In[40]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_stand = sc.fit_transform(X_data)


# In[41]:


vif = pd.DataFrame()
vif["Index"] = X_data.keys()
vif["Score"] = [variance_inflation_factor(x_stand,i) for i in range(len(X_data.keys()))]
vif.set_index(["Index"] , inplace = True)
vif


# #### We can see program type T and Y have high VIF, Hence dropping program type Y 

# In[42]:


X_data = Train.drop(["is_pass" , "program_type_Y"],axis=1)


# In[43]:


x_stand = sc.fit_transform(X_data)

vif = pd.DataFrame()
vif["Index"] = X_data.keys()
vif["Score"] = [variance_inflation_factor(x_stand,i) for i in range(len(X_data.keys()))]
vif.set_index(["Index"] , inplace = True)
vif


# #### Multicollinearity has been resolved

# # MODEL BUILDING

# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score , GridSearchCV , RepeatedStratifiedKFold
from sklearn.feature_selection import RFE 
from sklearn.metrics import roc_auc_score , roc_curve , classification_report , accuracy_score


# ### Logistic Regression

# In[45]:


lr = LogisticRegression()
lr.fit(X_data,Y_data)


# In[46]:


Y_predict = lr.predict(X_data)


# In[47]:


fpr , tpr , thresolds = roc_curve(Y_data,lr.predict_proba(X_data)[:,1])


# In[48]:


plt.figure()
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.show()


# In[49]:


print("The ROC_AUC score of training data is {:.2f}".format(roc_auc_score(Y_data,Y_predict)*100),"%")


# In[50]:


print("The average cross-validation score is {:.2f}".format(cross_val_score(lr,X_data,Y_data,cv=10).mean()*100))


# In[51]:


print("The accuracy score of training data is {:.2f}".format(accuracy_score(Y_data,Y_predict)*100),"%")


# In[52]:


print('The classification report is as follows:\n',classification_report(Y_data,Y_predict))


# In[53]:


print( np.unique( Y_predict ) )


# #### As we can clearly see that the dataset is imbalanced, hence resampling is done with replacement

# # RESAMPLING

# In[54]:


from sklearn.utils import resample
data_majority = Train[Train["is_pass"] == 1]
data_minority = Train[Train["is_pass"] == 0]


# In[55]:


data_minority_resampled = resample(data_minority , n_samples = 35606 , replace = True , random_state = 42)


# In[56]:


df = pd.concat([data_majority , data_minority_resampled])


# In[57]:


X = df.drop(["is_pass" , "program_type_Y"],axis=1)
Y = df["is_pass"]


# In[58]:


Y.value_counts()


# ### Logistic Regression on upsampled dataset

# In[59]:


lr.fit(X,Y)
Y_predict_sampled = lr.predict(X)


# In[60]:


print("The accuracy score of training data is {:.2f}".format(accuracy_score(Y,Y_predict_sampled)*100),"%")


# In[61]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[62]:


scores = cross_val_score(lr, X , Y, cv=cv, n_jobs=-1)


# In[63]:


print("The average cross-validation score is {:.2f}".format(scores.mean()*100),"%")


# In[64]:


print('The classification report is as follows:\n',classification_report(Y,Y_predict_sampled))


# In[65]:


print( np.unique( Y_predict_sampled ) )


# ### Random Forest Classifier(On imbalanced dataset)

# In[66]:


rfc = RandomForestClassifier(n_estimators=200, max_features = 10 , random_state=42)
rfc.fit(X_data,Y_data)


# In[67]:


Y_predict1 = rfc.predict(X_data)


# In[68]:


print("The ROC_AUC score of training data is {:.2f}".format(roc_auc_score(Y_data,Y_predict1)*100),"%")


# In[69]:


print("The average cross-validation score is {:.2f}".format(cross_val_score(rfc,X_data,Y_data,cv=10).mean()*100),"%")


# In[70]:


print('The classification report is as follows:\n',classification_report(Y_data,Y_predict1))


# In[71]:


data_frame = pd.DataFrame()
data_frame['importance'] = rfc.feature_importances_ * 100
data_frame['features'] = X_data.columns
data_frame.set_index('features' , inplace = True)
data_frame.sort_values(by = ["importance"] , ascending = True , inplace = True)
data_frame.importance.plot(kind = 'barh' , figsize = (11,6) , color = 'orange')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Features Vs. Importance')
plt.show()


# ### Self-Analysed Variables for Random Forest Classifier(On imbalanced dataset)

# In[72]:


X_data.columns


# In[73]:


X_new = X_data.drop(["education_Matriculation" , "program_type_U" , "program_type_T" , "program_type_V" , 
                    "difficulty_level_hard" , "difficulty_level_vary hard" , "program_type_X" , "program_type_Z",
                    "education_No Qualification" , "education_Masters"] , axis = 1)


# In[74]:


rfc1 = RandomForestClassifier(n_estimators=200 , random_state=42)
rfc1.fit(X_new,Y_data)
Y_predict11 = rfc1.predict(X_new)


# In[75]:


print("The ROC_AUC score of training data is {:.2f}".format(roc_auc_score(Y_data,Y_predict11)*100),"%")


# In[76]:


print('The classification report is as follows:\n',classification_report(Y_data,Y_predict11))


# ### RFE-based Variables for  Random Forest Classifier(on imbalanced dataset)

# In[82]:


rfe = RFE(rfc1 , 10)
rfe.fit(X_data , Y_data)


# In[83]:


new_columns = X_data.columns[rfe.support_]
new_columns


# In[86]:


rfc1.fit(X_data[new_columns],Y_data)
Y_predict2 = rfc1.predict(X_data[new_columns])


# In[87]:


print("The ROC_AUC score is {:.2f}".format(roc_auc_score(Y_data,Y_predict2)*100),"%")


# In[88]:


print('The classification report is as follows:\n',classification_report(Y_data,Y_predict2))


# ## Kneighbors Classifier(On balanced dataset)

# In[89]:


grid1 = {'n_neighbors': np.array([5,10,15,20]) , 'p' : np.array([3,6,9,12])}
knn = KNeighborsClassifier(metric='minkowski')
knn1 = GridSearchCV(knn, grid1, cv = 5)


# In[90]:


knn1.fit(X,Y)
y_predict_knn = knn1.predict(X)


# In[91]:


print("Tuned hyperparameter : {}".format(knn1.best_params_))


# In[66]:


## USING THE ABOVE COMPUTED HYPER-PARAMETRE

knn0 = KNeighborsClassifier(n_neighbors = 5 , p = 9 , metric='minkowski')
knn0.fit(X,Y)
y_predict_knn_new = knn0.predict(X)


# In[67]:


print("The accuracy score of the model is {:.2f}".format(accuracy_score(Y , y_predict_knn_new) * 100) , "%")


# In[68]:


print('The classification report is as follows:\n',classification_report(Y , y_predict_knn_new))


# # CONCLUSION

# In[95]:


print("The ROC_AUC score with Logistic regression on imbalanced data is {:.2f}".format(roc_auc_score(Y_data,Y_predict)*100),"%")
print()
print("The accuracy score with Logistic regression on balanced data is {:.2f}".format(accuracy_score(Y,Y_predict_sampled)*100),"%")
print()
print("The ROC_AUC score with Random forest classifier on imbalanced data is {:.2f}".format(roc_auc_score(Y_data,Y_predict1)*100),"%")
print()
print("The ROC_AUC score with selecting best features with random forest classifier is {:.2f}".format(roc_auc_score(Y_data,Y_predict11)*100),"%")
print()
print("The ROC_AUC score with RFE-variables with Random forest classifier is {:.2f}".format(roc_auc_score(Y_data,Y_predict2)*100),"%")
print()
print("The accuracy score with KNeighbors classifier on balanced data is {:.2f}".format(accuracy_score(Y , y_predict_knn_new) * 100) , "%")


# #### We can observe from the entire analysis that Random forest classifier has performed well with maximum features as 10 selected 
# #### in an arbitrary manner  and with best selected features and RFE-features also Random forest classifier has performed well but 
# #### "KNN" algorithm's performance on a balanced dataset is better. So,"KNN" has finally been selected for test data prediction.

# # Test Data Prediction

# In[99]:


Test1 = Test.drop(["program_type_Y"],axis=1)
y_test_predict = knn0.predict(Test1)
test['is_pass'] = y_test_predict
test.head()


# ### Saving the prediction file

# In[100]:


test.to_csv("Final test data.csv",index=False )


# ### Model Deployment

# In[69]:


import pickle


# In[79]:


file = open('model.pkl' , 'wb')
pickle.dump(knn0,file)

