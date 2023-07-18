import pandas as pd
import numpy as np

bdd = pd.read_csv('customer.csv')

# 1- suppression des colonnes inutiles
bdd.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# 2- No encodage des variables catégorielles pour utilser le smote-nc
new_nc = bdd.iloc[:20]
new_nc.to_csv('xgboost/Smote_NC/new_nc.csv', index=False)
work_nc = bdd.iloc[20:]
work_nc.to_csv('xgboost/Smote_NC/work_nc.csv', index= False)

# 3- Encodage des variables catégorielles 
# Les variables catégorielles "Geography" et "Gender" doivent être encodées en variables numériques pour être utilisées dans les modèles de prédiction

mapping_geography = {'France': 0, 'Germany': 1, 'Spain': 2}
mapping_gender = {'Female': 0, 'Male': 1}
bdd['Geography'] = bdd['Geography'].replace(mapping_geography)
bdd['Gender'] = bdd['Gender'].replace(mapping_gender)

# Création d'une nouvelle donnée pour un futur test
new_data = bdd.iloc[:20]
new_data.to_csv('xgboost/Smote/new_data.csv', index=False)
work_data = bdd.iloc[20:]
work_data.to_csv('xgboost/Smote/work_data.csv', index= False)