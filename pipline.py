import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler




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
    # list_columns = []
    # for col in df.columns:
    #     if df[col].dtypes == 'object':
    #         list_columns.append(col)
    categoric_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'Churn']
    
    for col in categoric_cols:
        df[col] = le.fit_transform(df[col])
    return df
data_clean = pretraitement_variable(data)


numeric_cols =data[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
clean_colomns = pd.concat([data_clean,numeric_cols], axis=0)
print(clean_colomns)



def split_features(df):
    
# 1- separation de features et la cible(churn)
    x = df.drop('Churn', axis=1)
    y = df['Churn']
    return x, y


def split_data(x,y):

# 1- split_train_test 
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    return  x_train,x_test,y_train,y_test


def normalized_data(x_test,x_train):

# 1- normalisation des variables numerique
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)
    x_train = scaler.transform(x_train)
    return x_test,x_train















    





