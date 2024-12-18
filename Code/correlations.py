import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier

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

# Calcul des corrélations initiales entre attributs et labels
correlations = features.corr(method='pearson')
print(correlations)

# Visualiser avec une heatmap
fig = plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation entre les variables")
fig.savefig("../Correlations/correlations_initiales")

print("Corrélations initiales entre attributs et labels :\n", correlations)

models_from_joblib = {
    "RandomForest": joblib.load("../joblibs/RandomForest_BestModel_08159.joblib"),
    "AdaBoost": joblib.load("../joblibs/AdaBoost_BestModel_08038.joblib"),
    "Stacking": joblib.load("../joblibs/Stacking_BestModel_08194.joblib")
}
correlations_model = {}

for name, model in models_from_joblib.items():
    # Prédictions sur les données de test
    y_pred_test = model.predict(scaled_df_test)

    # Calcul des corrélations entre les attributs de test et les prédictions
    correlations_model[name] = scaled_df_test.corrwith(
        pd.Series(y_pred_test, index=scaled_df_test.index),
        method='pearson'
    )

correlations_model_df = pd.DataFrame(correlations_model)

# Créer des graphiques pour chaque modèle
for model_name in correlations_model_df.columns:
    sorted_correlations = correlations_model_df[model_name].sort_values(ascending=False)
    fig = plt.figure(figsize=(10, 6))
    sorted_correlations.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Corrélations entre les variables et les prédictions du modèle {model_name}")
    plt.xlabel("Variables")
    plt.ylabel("Coefficient de corrélation")
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout()
    fig.savefig(f"../Correlations/correlations_{model_name}")
    plt.show()