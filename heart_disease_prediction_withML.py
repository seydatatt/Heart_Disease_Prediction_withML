# -*- coding: utf-8 -*-
#import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

import warnings
warnings.filterwarnings("ignore")
#load dateset and EDA 
data = pd.read_csv("heart_disease_uci.csv")
data = data.drop(columns = ["id"])
data.info()
describe = data.describe()

numeric_features = data.select_dtypes(include = [np.number]).columns.tolist() 

plt.figure()
sns.pairplot(data, vars= numeric_features, hue = "num")
plt.show()

plt.figure()
sns.countplot(x = "num", data = data)
plt.show()

#handing missing value
data.isnull().sum()
data = data.drop(columns = ["ca"])
print(data.isnull().sum())

data["trestbps"].fillna(data["trestbps"].median(),inplace = True) 
data["chol"].fillna(data["chol"].median(),inplace = True) 
data["fbs"].fillna(data["fbs"].mode()[0],inplace = True) 
data["restecg"].fillna(data["restecg"].mode()[0],inplace = True) 
data["thalch"].fillna(data["thalch"].median(),inplace = True) 
data["exang"].fillna(data["exang"].mode()[0],inplace = True) 
data["oldpeak"].fillna(data["oldpeak"].median(),inplace = True)
data["slope"].fillna(data["slope"].mode()[0],inplace = True) 
data["thal"].fillna(data["thal"].mode()[0],inplace = True) 

print(data.isnull().sum())

#train test split 

X = data.drop(["num"], axis = 1 )
y = data["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
categorical_features = ["sex", "dataset", "cp","restecg","exang","slope","thal"]
numeric_features = ["age","trestbps","chol","fbs","thalch","oldpeak"] 

X_train_num = X_train[numeric_features]
X_test_num = X_test[numeric_features]

scaler = StandardScaler()   
X_train_num_scaled= scaler.fit_transform(X_train_num)
X_test_num_scaled= scaler.fit_transform(X_test_num)

encoder = OneHotEncoder(sparse_output =False, drop = "first")
X_train_cat = X_train[categorical_features]
X_test_cat = X_test[categorical_features]

X_train_cat_encoded = encoder.fit_transform(X_train_cat)
X_test_cat_encoded = encoder.fit_transform(X_test_cat)

X_train_transformed = np.hstack((X_train_num_scaled, X_train_cat_encoded))
X_test_transformed = np.hstack((X_test_num_scaled, X_test_cat_encoded))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()
voting_clf = VotingClassifier(estimators=[
    ("rf", rf),
    ("knn",knn)
    ], voting = "soft")

#model training 
voting_clf.fit(X_train_transformed, y_train)

# make a predict with test data 
y_pred = voting_clf.predict(X_test_transformed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report: ")
print(classification_report(y_test, y_pred))

#Confusion Matrix Figure 

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues",cbar = False) 
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

