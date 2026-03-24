# Preparation des données avant l'entrainement des modèles

# importation des libraries :
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Objectifs :
# 
# - Normaliser les noms de colonnes
# - Séparer la colonne cible des features
# - Supprimer les colonnes inutiles 
# - Identifier les colonnes numériques et catégorielles
# - Utiliser un préprocesseur pour mettre les données à la bonne forme
# - Divise les données en 80% train / 20% test en conservant l'équilibre des classes

def prepare_data(df: pd.DataFrame, target_col: str = 'legendary'):

    df.columns = df.columns.str.lower().str.replace(' ', '_') #mettre les noms de colonnes en minuscules et remplacer les espaces par des underscores

    COLS_TO_DROP = ['number', 'name', 'abilities', 'mega_evolution', 'alolan_form', 'galarian_form', 'bmi'] #colonnes à supprimer

    # On sépare la colone cible des autres colonnes (features)
    y = df[target_col] #colonne cible
    X = df.drop(columns=[target_col] + COLS_TO_DROP)#features

# prétraitement des données numériques et catégorielles
    cat_cols = X.select_dtypes(include='object').columns 
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),           
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

# séparation des données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, preprocessor
