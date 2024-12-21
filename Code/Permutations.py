from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



features = pd.read_csv("../Dataset/features.csv")
label = pd.read_csv("../Dataset/labels.csv")

X_train, X_test, y_train, y_test= train_test_split(
    features, label, test_size=0.2, random_state=0)


scaler = StandardScaler()

scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)

scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
scaled_df_test = pd.DataFrame(scaled_data_test, columns=X_test.columns)

scaled_df_train.to_csv('scaled_train_features.csv', index=False)
scaled_df_test.to_csv('scaled_test_features.csv', index=False)

models_from_joblib = {
    "RandomForest": joblib.load("../joblibs/RandomForest_BestModel_08159.joblib"),
    "GradientBoost": joblib.load("../joblibs/GradientBoost_BestModel_08158.joblib"),
    "AdaBoost": joblib.load("../joblibs/AdaBoost_BestModel_08038.joblib"),
    "Stacking": joblib.load("../joblibs/Stacking_BestModel_08194.joblib")
}

# Calculer l'importance des attributs pour chaque modèle
feature_importances = {}
for name, model in models_from_joblib.items():
    # Importance par permutation
    perm_importance = permutation_importance(
        model, scaled_df_test, y_test.values.ravel(), scoring='accuracy', n_repeats=10, random_state=0
    )
    feature_importances[name] = pd.Series(
        perm_importance.importances_mean, index=scaled_df_test.columns
    ).sort_values(ascending=False)

# Afficher les graphes des importances des attributs
for model_name, importances in feature_importances.items():
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Importance des attributs pour le modèle {model_name}")
    plt.xlabel("Attributs")
    plt.ylabel("Importance moyenne (Permutation)")
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

