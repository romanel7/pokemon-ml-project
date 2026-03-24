# Prédiction des Pokémon Légendaires

> Projet Data Science — ESIGELEC | Mars 2026  
> **Clarra Masse & Romane Lesueur**

---

## Problématique

**Peut-on prédire automatiquement si un Pokémon est légendaire à partir de ses caractéristiques ?**

Les Pokémon légendaires représentent environ 12% du dataset, ce qui en fait un problème de classification déséquilibrée. L'objectif est de comparer plusieurs modèles prédictifs pour identifier celui qui détecte le mieux cette classe minoritaire.

---

## 📁 Structure du projet

```
├── data/
│   ├── raw/                        # Données brutes originales
│   └── processed/                  # Données nettoyées
│       └── pokemon_clean.csv
│
├── notebooks/
│   ├── 1_exploration_nettoyage.ipynb   # EDA + nettoyage
│   └── 2_modelisation/
│       ├── 1_Random_Forest.ipynb
│       ├── 2_XGBoost.ipynb
│       └── 3_SVC-RBF.ipynb
│
├── Prepro.py                        # Fonction centralisée de préprocessing
└── README.md
```

---

## Préprocessing (`Prepro.py`)

Tous les notebooks de modélisation partagent une fonction commune `prepare_data()` :

- Nettoyage des noms de colonnes
- Suppression des colonnes non pertinentes (`name`, `abilities`, `bmi`...)
- Séparation features / cible
- `StandardScaler` sur les variables numériques
- `OneHotEncoder` sur les variables catégorielles
- Split train/test stratifié (80/20)

---

## Modèles comparés

| Modèle | Accuracy | F1 Légendaires | Recall Légendaires |
|---|---|---|---|
| Random Forest | 99% | 94% | 92% |
| Gradient Boosting | 99% | 96% | 96% |
| SVC-RBF | — | — | — |

---

## Résultats clés

- Les trois modèles atteignent une **accuracy de 99%** malgré le déséquilibre du dataset
- Le **catch_rate** est la variable la plus discriminante dans tous les modèles
- Le Gradient Boosting s'appuie quasi exclusivement sur le catch_rate (70% d'importance), tandis que le Random Forest répartit l'importance sur plusieurs variables

---

## Lancer le projet

### Prérequis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Ordre d'exécution recommandé

```
1. notebooks/1_exploration_nettoyage.ipynb
2. notebooks/2_modelisation/1_Random_Forest.ipynb
3. notebooks/2_modelisation/2_XGBoost.ipynb
4. notebooks/2_modelisation/3_SVC-RBF.ipynb
```

---

## Source des données

Dataset Pokémon issu de [Kaggle](https://www.kaggle.com/) enrichi de statistiques de jeu (catch rate, courbe d'expérience, stats de combat).