import numpy as np
import joblib
import pandas as pd

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



models = {
    "RandomForest":RandomForestClassifier(),
    "AdaBoost":AdaBoostClassifier(algorithm="SAMME"),
    "GradientBoost":GradientBoostingClassifier(),
    "Stacking":StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier()),
            ("ab", AdaBoostClassifier(algorithm="SAMME")),
            ("gb", GradientBoostingClassifier())
        ],
        final_estimator=LogisticRegression()
    )
}

param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1, 2]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10]
    },
    "Stacking": {
        "final_estimator__C": [0.1, 1, 10] 
    }
}
def optimize_model(name, model, param_grid, X_train, y_train):
    print(f"Optimisation des hyperparamètres pour {name}...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(scaled_data_train, y_train.values.ravel())  # Utiliser y_train.ravel() si nécessaire
    print(f"Meilleurs paramètres pour {name} : {grid_search.best_params_}")
    print(f"Meilleure score pour {name} : {grid_search.best_score_}")
    return grid_search.best_estimator_

# Optimiser les modèles
optimized_models = {}
for name, model in models.items():
    if name in param_grids:
        optimized_models[name] = optimize_model(name, model, param_grids[name], scaled_df_train, y_train)

results={}
for name, model in optimized_models.items():
    y_pred_train = cross_val_predict(model, scaled_df_train, y_train.values.ravel())
    model.fit(scaled_df_train, y_train.values.ravel())
    y_pred_test = model.predict(scaled_df_test)

    metrics = {
        "accuracy_train": accuracy_score(y_train, y_pred_train),
        "accuracy_test": accuracy_score(y_test, y_pred_test),
        "classification_report_train": classification_report(y_train, y_pred_train),
        "classification_report_test": classification_report(y_test, y_pred_test),
        "confusion_matrix_train": confusion_matrix(y_train, y_pred_train),
        "confusion_matrix_test": confusion_matrix(y_test, y_pred_test),
    }
    print(name)
    print("-------------------------")
    print(metrics)
    print("-------------------------")

    results[name] = metrics

    print(f"Accuracy (train): {metrics['accuracy_train']}")
    print(f"Accuracy (test): {metrics['accuracy_test']}")
    print("\nClassification Report (test):\n", metrics['classification_report_test'])
    print("\nConfusion Matrix (test):\n", metrics['confusion_matrix_test'])

summary = []
for name, metrics in results.items():
    summary.append({
        "Model": name,
        "Accuracy (train)": metrics["accuracy_train"],
        "Accuracy (test)": metrics["accuracy_test"],
    })

summary_df = pd.DataFrame(summary)
print("\nRésumé des performances :\n", summary_df)
    

for name, model in optimized_models.items():
    joblib.dump(model, f"{name}_optimized_model.joblib")
