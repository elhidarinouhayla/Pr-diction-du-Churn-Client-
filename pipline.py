import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report




# 1- charger les donnees
def charge_donnees():
    file = pd.read_csv("telecom_churn.csv")
    return file
data= charge_donnees()

def pretraitement_variable(df):
    df = charge_donnees()
# 1- Convertir la colonne 'TotalCharges' en numérique
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')

# 2- Remplacer les valeurs manquantes dans 'TotalCharges' par la moyenne
    moyenne = df['TotalCharges'].mean()
    df['TotalCharges'] = df['TotalCharges'].fillna(moyenne)

     
# 3- Supprimer les colonnes inutiles
    df = df.drop(columns=['customerID', 'gender'])

# 4- Encoder les variables catégorielles
    list_columns = []
    for col in df.columns:
        if df[col].dtypes == 'object':
            list_columns.append(col)
    print(list_columns)
    df[col] = le.fit_transform(df[col])
    return df

data = pretraitement_variable(data)


def split_features(df):
    
# 1- separation de features et la cible(churn)
    x = df.drop('Churn', axis=1)
    y = df['Churn']
    return x, y

def split_data(x,y, test_size=0.2):

# 1- split_train_test 
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    return  x_train,x_test,y_train,y_test

# def scaler_data(x_test,x_train):
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     return x_train_scaled, x_test_scaled











    





