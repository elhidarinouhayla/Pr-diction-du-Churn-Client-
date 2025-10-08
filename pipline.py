import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split



# 1- charger les donnees
def charge_donnees():
    file = pd.read_csv("telecom_churn.csv")
    return file
data= charge_donnees("telecom_churn.csv")

def transforme_variable(df):
    df = charge_donnees()
    list_columns=[]
# 1- Convertir la colonne 'TotalCharges' en numérique
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce')

# 2- Remplacer les valeurs manquantes dans 'TotalCharges' par la moyenne
    moyenne = df['TotalCharges'].mean()
    df['TotalCharges'] = df['TotalCharges'].fillna(moyenne)

     
# 3- Supprimer les colonnes inutiles
    df = df.drop(columns=['customerID', 'gender'])

# 4- Encoder les variables catégorielles
    for col in df.columns:
        if df[col].dtypes == 'object':
          list_columns.append(col)
    print(list_columns)
    df['list_columns'] = le.fit_transform(df['list_columns'])

nettoyage = transforme_variable(data)


def split_features(df):
    
    x = df.drop('Churn')
    y = df['Churn']
    x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2)
    print("x_train: ", x_train)
    print("x_test: ", x_test)
    print("y_train: ", y_train)
    print("y_test:", y_test)







    





