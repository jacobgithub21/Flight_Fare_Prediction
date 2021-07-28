#!/usr/bin/env python
# coding: utf-8

# # The dataset is provided with prices of flight tickets for various airlines between the months of March and June of 2019 and between various cities and the objective of this task is to predict the flight ticket price.
# 
# 
# 
# 1)  Apply Suitable Machine Learning Algorithms for given the dataset and build a machine learning model that can accurately predict the price of a product based on the given factors.
# 
# 2)  Apply Suitable Data Cleaning Techniques for the data set and transform the data.
# 
# 3)  Perform Extensive Exploratory Data Analysis on the dataset and Share your findings.
# 
# 4)  Apply Suitable Evaluation Methods for your Machine Learning Model.
# 
# 5)  Note: Need to solve the above task by using atleast minimum two Machine Learning Algorithms and compare the results of all the models.

# In[ ]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Train Data

# In[260]:


train_data= pd.read_excel('D://data.xlsx')


# In[261]:


pd.set_option('display.max_columns',None) 


# In[262]:


train_data.head()


# In[263]:


train_data.shape


# In[264]:


train_data.info()


# In[265]:


# Looking ate the unique values of Categorical Features
print(train_data['Airline'].unique())
print(train_data['Destination'].unique())
print(train_data['Source'].unique())


# In[266]:


train_data.describe(include = 'all')
# Price is the only integer column
# all other column are string/object type


# In[267]:


train_data['Duration'].value_counts()


# In[268]:


# Dropping the NaN values
train_data.dropna(inplace=True)


# In[269]:


train_data.isnull().sum()


# In[270]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar = False,cmap='viridis')


# # Feature Engineering

# In[272]:


# extracting information from 'date_of_journey' column and storing in new columns 'Journey_month' and 'journey_day'
train_data['Journey_month']= pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.month


# In[273]:


train_data['Journey_day']= pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.day


# In[14]:


train_data.head()


# In[274]:


train_data.drop(["Date_of_Journey"],axis=1,inplace=True)


# In[275]:


# Departure time 

# Extracting Hours from 'Dep_Time' column by creating a new column 'Dep_hour'
train_data["Dep_hour"]= pd.to_datetime(train_data['Dep_Time']).dt.hour

# Extracting Minutes from 'Dep_Time' column by creating a new column 'Dep_min'
train_data["Dep_min"]= pd.to_datetime(train_data['Dep_Time']).dt.minute

# drop Dep_time columns 
train_data.drop(["Dep_Time"],axis=1,inplace=True)


# In[276]:


train_data.head()


# In[277]:



#'Arrival_time' column 

# Extracting Hours from 'Arrival_Time' column by creating a new column 'Arr_hour'
train_data["Arr_hour"]= pd.to_datetime(train_data['Arrival_Time']).dt.hour

# Extracting Minutes from 'Arrival_Time' column by creating a new column 'Arr_min'
train_data["Arr_min"]= pd.to_datetime(train_data['Arrival_Time']).dt.minute

#drop Arrival_time columns
train_data.drop(["Arrival_Time"],axis=1,inplace=True)


# In[278]:


train_data.head()


# In[289]:


plt.figure(figsize = (15, 10))
plt.title('Count of flights month wise')
graph=sns.countplot(x = 'Journey_month', data = train_data)
plt.xlabel('Month')
plt.ylabel('Count of flights')
for p in graph.patches:   #print the text label in a particular position in the chart.
    graph.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')


# In[292]:


plt.figure(figsize = (15, 10))
plt.title('Count of flights with different Airlines')
graph=sns.countplot(x = 'Airline', data =train_data)
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
for p in graph.patches:
    graph.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')


# In[ ]:


#There are more number of flights of Jet Airways.
#Jet Airways Business, Vistara Premium economy, Trujet have actually almost negligible flights.


# In[279]:


#Duration
# It is the difference between Departure and Arrial time

duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[280]:


# adding duration_hours and duration_mins to our dataframe
train_data['Duration_hours']= duration_hours
train_data['Duration_mins']= duration_mins


# In[281]:


#drop the 'Duration' column 
train_data.drop(['Duration'],axis=1,inplace=True)


# In[282]:


train_data.head()


# # HANDLING CATEGORICAL DATA

# In[283]:


train_data['Airline'].value_counts()


# In[294]:


# AIRLINE vs PRICE
plt.figure(figsize = (15, 10))
plt.title('Price VS Airlines')
plt.scatter(train_data['Airline'], train_data['Price'])
plt.xticks(rotation = 90)
plt.xlabel('Airline')
plt.ylabel('Price of ticket')
plt.xticks(rotation = 90)


# In[284]:


# AIRLINE vs PRICE
sns.catplot(y='Price',x='Airline',data= train_data.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show


# In[285]:


#using boxplot
plt.figure(figsize=(29, 7))
sns.boxplot(x='Airline', y='Price', data = train_data,palette = 'winter')
plt.show()


# In[ ]:


# From above diagram Jet airways Business have the highest price &  apart from the first airline almost all are having similarmedian


# In[286]:


# As Airline column has nominal Categorical data , we will perform One Hot encoding
Airline=train_data[["Airline"]]
Airline= pd.get_dummies(Airline,drop_first=True)
Airline.head()


# In[26]:


train_data['Source'].value_counts()


# In[27]:


# Source vs PRICE
sns.catplot(y='Price',x='Source',data= train_data.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show


# In[29]:


# as Source column has  nominal categorical data, we will perform OneHotEncoding

Source=train_data[["Source"]]
Source= pd.get_dummies(Source,drop_first=True)
Source.head()


# In[30]:


train_data['Destination'].value_counts()


# In[31]:


# Destination vs PRICE
sns.catplot(y='Price',x='Destination',data= train_data.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show


# In[32]:


# as Destination column has  nominal categorical data, we will perform OneHotEncoding

Destination=train_data[["Destination"]]
Destination= pd.get_dummies(Destination,drop_first=True)
Destination.head()


# In[33]:


train_data['Route']


# In[34]:


train_data.drop(["Route","Additional_Info"],axis=1,inplace=True)


# In[35]:


train_data.head()


# In[36]:


train_data["Total_Stops"].value_counts()


# In[37]:


# As total_stops column hs Ordinal Categorical type of data, So we perform Label Encoding

train_data.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)
train_data.head()


# In[38]:


# Concatenate dataframe --> train_data + airline + source and destination
data_train= pd.concat([train_data,Airline,Source,Destination],axis=1)
data_train.head()


# In[39]:


# dropping the unnecessary columns now
data_train.drop(["Airline","Source","Destination"],axis=1,inplace=True)
data_train.head()


# In[40]:


data_train.shape


# # TEST DATA

# In[57]:


# we are not combining test and train data to prevent the Data Leakage.
test_data= pd.read_excel('D://data.xlsx')
test_data = test_data.loc[0:2672,'Airline':'Additional_Info']
test_data.head()


# In[58]:


test_data.shape


# In[59]:


# performing  all the steps again for the test data.

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[60]:


data_test.head()


# # Feature Selection
# #### Finding out the best features which will contribute and have good relation with our Target variable
# Following are some feature selection methods:
# 1) heatmap
# 2) feature_importance_
# 3) SelectKBest

# In[61]:


data_train.shape


# In[62]:


data_train.columns


# In[65]:


# making X our independent variable
X= data_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arr_hour', 'Arr_min', 'Duration_hours', 'Duration_mins',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[63]:


# y will be our dependent feature
y= data_train.iloc[:,1]
y.head()


# In[246]:


train_data.corr()
#its shows the direction of the  magnitude, like if we have +ve value our two features will be in increasing condition,else decreasing condition. 


# In[64]:


# Finding correlation between Independent and Dependent Feature
# it find the interrelationship between the 2 features, for this condition our features will be in int or float type for finding the correlation

plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')

plt.show()


# Extreme green means highly correlated, 
# Extreme red means negatively correlated.

# If two independent features are highly correlated , then we can drop any one of them as both are doing almost same task.

# In[66]:


# To know the Important features using ExtraTreeRegressor 

from sklearn.ensemble import ExtraTreesRegressor
selection= ExtraTreesRegressor()
selection.fit(X,y)


# In[67]:


# looking at important features given bt ExtraTreesRegressor
print(selection.feature_importances_)


# In[68]:


#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # MODEL BUILDING

# In[182]:


# training testing and splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # KNeighborsRegressor

# In[204]:


from sklearn.neighbors import KNeighborsRegressor


# In[205]:


k_reg = KNeighborsRegressor()
k_reg.fit(X_train,y_train)


# In[206]:


# prediction variable 'y_pred'
y_pred = k_reg.predict(X_test)


# In[207]:


# Accuracy to training sets
k_reg.score(X_train,y_train)


# In[208]:


# Accuracy to test sets
k_reg.score(X_test,y_test)


# In[209]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[210]:


# R square error
metrics.r2_score(y_test,y_pred)


# # XGBRegressor

# In[190]:


from xgboost import XGBRegressor


# In[191]:


xgb = XGBRegressor()
xgb.fit(X_train, y_train)


# In[192]:


# prediction variable 'y_pred'
y_pred= tree.predict(X_test)


# In[193]:


# Accuracy to training sets
tree.score(X_train,y_train)


# In[194]:


# accuracy of Testing sets
tree.score(X_test,y_test)


# In[195]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[196]:


# R square error
metrics.r2_score(y_test,y_pred)


# # DecisionTreeRegressor

# In[197]:


from sklearn.tree import DecisionTreeRegressor


# In[198]:


tree =DecisionTreeRegressor ()
tree.fit(X_train, y_train)


# In[199]:


# prediction variable 'y_pred'
y_pred= tree.predict(X_test)


# In[200]:


# Accuracy to training sets
tree.score(X_train,y_train)


# In[201]:


# accuracy of Testing sets
tree.score(X_test,y_test)


# In[202]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[203]:


# R square error
metrics.r2_score(y_test,y_pred)


# # Random Forest Regressor

# In[106]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[107]:


# prediction variable 'y_pred'
y_pred= reg_rf.predict(X_test)


# In[108]:


# Accuracy to training sets
reg_rf.score(X_train,y_train)


# In[109]:


# accuracy of Testing sets
reg_rf.score(X_test,y_test)


# In[110]:


sns.distplot(y_test-y_pred)
plt.show()


# The above plot is showing Gaussian distribution which shows that our predictions are good

# In[111]:


plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# The linear distribution in the above scatter plot shows that our predictions are good

# In[112]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[113]:


# R square error
metrics.r2_score(y_test,y_pred)
                


# In[295]:


### To achive more accuracy we go for hyperparameter tunig method 


# # # Hyperparameter Tuning
# ###### There are two techniques of Hyperparameter tuning i.e 
# 1) RandomizedSearchCv
# 2) GridSearchCV
# ##### We use RandomizedSearchCv because it is much faster than GridSearchCV

# In[216]:


from sklearn.model_selection import RandomizedSearchCV


# In[218]:


# Randomized Search CV
# Number of trees in random forest
n_estimators = [100, 200, 300, 400, 500]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]


# In[219]:


# create random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[233]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_absolute_error', n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs =- 1)


# In[234]:


rf_random.fit(X_train,y_train)


# In[235]:


# looking at best parameters
rf_random.best_params_


# In[236]:


prediction = rf_random.predict(X_test)


# In[237]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# Gaussian distribution shows our predictions are very good

# In[238]:


# plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[239]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[240]:


metrics.r2_score(y_test,prediction)


# In[296]:


# we get some incresese in accuracy score..(2%)


# #  predictive_score :                       training sets          ,                             test sets 
# 
# #  KNeighborsRegressor                  0.7353                  ,                             0.5743
# #  XGBRegressor                               0.9692                  ,                             0.7309
# #  DecisionTreeRegressor                 0.969248              ,                             0.72687
# #  Random Forest Regressor            0.952826              ,                             0.81600
# 
# # By seeing the accuracy of the  above models we knowing Random Forest Regressor gives good fit compared to other regressors, and the MSE,MAE,RMSE is low comapred to other models.
# 

# In[ ]:




