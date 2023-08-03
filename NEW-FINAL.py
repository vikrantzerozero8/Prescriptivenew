# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:29:30 2023

@author: aivam
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import sweetviz
from sqlalchemy import create_engine
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


# Credentials to connect to Database
user = 'root'  # user name
pw = 'newpasswords'  # password
db = 'credit_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


# Import file from SQL 

from sqlalchemy import text
conn = engine.connect()
query = text('SELECT * FROM creditc')


data = pd.read_sql_query(query, conn)

#  EDA & VISUALZATION

data.describe()
data.info()
data.mean()



# Checking the distribution of Output

plt.figure(figsize=(8, 6))
sns.countplot(data['creditScore'])
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Count')
plt.show()

# checking any relationship between the credit score and purpose of credit 

plt.figure(figsize=(12, 6))
sns.countplot(x='Cpur', hue='creditScore', data=data)
plt.title('Credit Score vs Purpose of Credit')
plt.xlabel('Purpose of Credit')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Credit Score')
plt.show()


plt.hist(data.Cbal, color = 'orange', edgecolor="Black")
plt.hist(data.Sbal, color = 'orange', edgecolor="Black")
plt.hist(data.MSG, color = 'orange', edgecolor="Black")
plt.hist(data.Cpur, color = 'orange', edgecolor="Black")

# Understanding the relationship between the inputs
data.corr()

# performing Auto EDA to understand the behaviour of the data 
import dtale
import pandas as pd
df = pd.read_csv(r"D:/creditc.csv")
d = dtale.show(df)
d.open_browser()

import sweetviz as sv
s = sv.analyze(data)
s.show_html()


from pandas_profiling import ProfileReport 
p = ProfileReport(data)
p
p.to_file("output.html")


# Data Preprocessing Pipeline

from feature_engine.outliers import Winsorizer

X = data.iloc[:,:19]
X['telephone'].value_counts()
X.drop(columns=['Ndepend'], inplace=True)
Y = data["creditScore"]
# Select numerical and categorical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64'])
categorical_features = X.select_dtypes(include=['object'])

# Create a transformer to one-hot encode the categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a transformer to winsorize the numerical columns
numerical_transformer = Pipeline(steps=[
    ('winsor', Winsorizer(capping_method='iqr', tail='both', fold=1.5))
])

# Create a preprocessor to apply the transformers to appropriate columns
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features.columns),
    ('num', numerical_transformer, numerical_columns.columns)
])

# Create the final pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('scaler', StandardScaler())  # Optionally add a scaler after preprocessing
])

# Fit the pipeline to the data
pipeline.fit(X)

# Save the pipeline to a joblib file
joblib.dump(pipeline, 'pipeline.joblib')


# Applying the Data Preprocessing by using Preprocessing Pipline 
pipeline = joblib.load('pipeline.joblib')
clean  = pipeline.transform(X)
clean_final = pd.DataFrame(clean, columns=pipeline.named_steps['preprocess'].get_feature_names_out())

X = clean_final

Y = data["creditScore"]

# Model Building 

# Partitiong the data 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state= 30)


# Initialize the logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# Train the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = logreg.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Calculate training accuracy
training_accuracy = logreg.score(X_train, y_train)

# Print the training accuracy
print("Training Accuracy:", training_accuracy)










param_grid = {
    'C': [0.8, 0.9,1,1.1,1.2,2,3],  # Inverse of regularization strength
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga']  # Optimization solver
}



# Initialize GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(logreg, param_grid, cv=5, n_jobs=-1)

# Train the model on the training data with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Use the best hyperparameters to create the final logistic regression model
logreg_best = LogisticRegression(**best_params)

# Train the final model on the training data with the best hyperparameters
logreg_best.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = logreg_best.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Best Hyperparameters:", best_params)
print("Testing Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Training Accuracy

logreg_best.fit(X_train, y_train)

# Calculate training accuracy
training_accuracy = logreg_best.score(X_train, y_train)

# Print the training accuracy
print("Training Accuracy:", training_accuracy)





# Hyper parameter

# Applying the RandomizedSearchCV
param_grid = {
    'C': [0.8, 0.9, 1, 1.1, 1.2, 2, 3],  # Inverse of regularization strength
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga']  # Optimization solver
}

# Finding the best parameters
random_search = RandomizedSearchCV(logreg, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)

# Train the model on Training data
random_search.fit(X_train, y_train)

# Finding best parameter
best_params = random_search.best_params_

# Best model selection
logreg_best = LogisticRegression(**best_params)

# Train the final model on the training data with the best hyperparameters
logreg_best.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = logreg_best.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
classification_rep_df = pd

# Print the results
print("Best Hyperparameters:", best_params)
print("Testing Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)

# Training Accuracy
logreg_best.fit(X_train, y_train)

# Calculate training accuracy
training_accuracy = logreg_best.score(X_train, y_train)



# Print the training accuracy
print("Training Accuracy:", training_accuracy)
pickle.dump(logreg_best, open('logic.pkl', 'wb'))



# New Data Prediction


model1 = pickle.load(open('logic.pkl', 'rb'))
pipe = joblib.load('pipeline.joblib')


new = pd.read_csv(r"D:/new-pred.csv")

new.drop(columns=['Ndepend'], inplace=True)



clean = pd.DataFrame(pipe.transform(new), columns = new.columns)

# Apply the same preprocessing steps as used for the training data
clean = pd.DataFrame(pipeline.transform(new), columns=pipeline.named_steps['preprocess'].get_feature_names_out())

prediction = pd.DataFrame(model1.predict(clean), columns = ['choice_pred'])

final = pd.concat([prediction, data], axis = 1)

final.to_csv('preic')

final.head()

