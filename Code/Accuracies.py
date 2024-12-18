import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


featuresN = pd.read_csv("../Dataset/featuresN.csv")
labelN = pd.read_csv("../Dataset/labelsN.csv")
featuresC = pd.read_csv("../Dataset/featuresC.csv")
labelC = pd.read_csv("../Dataset/labelsC.csv")
scaler = joblib.load("../joblibs/scaler.joblib")
models_from_joblib = {
    "RandomForest": joblib.load("../joblibs/RandomForest_BestModel_08159.joblib"),
    "GradientBoost": joblib.load("../joblibs/GradientBoost_BestModel_08158.joblib"),
    "AdaBoost": joblib.load("../joblibs/AdaBoost_BestModel_08038.joblib"),
    "Stacking": joblib.load("../joblibs/Stacking_BestModel_08194.joblib")
}


scaled_co = scaler.transform(featuresN)
scaled_df = pd.DataFrame(scaled_co, columns=featuresN.columns)

results={}
for name, model in models_from_joblib.items():
    # Prédictions sur les données de test
    y_pred_test = model.predict(scaled_df)
    metrics = {
    "accuracy_colorado": accuracy_score(labelN, y_pred_test),
    "classification_report_colorado": classification_report(labelN, y_pred_test),
    "confusion_matrix_colorado": confusion_matrix(labelN, y_pred_test),
    }
    print(name)
    print("-------------")
    print(f"Accuracy (test): {metrics['accuracy_colorado']}")
    print("\nClassification Report (test):\n", metrics['classification_report_colorado'])
    print("\nConfusion Matrix (test):\n", metrics['confusion_matrix_colorado'])