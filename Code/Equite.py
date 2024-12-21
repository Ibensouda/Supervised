import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

features = pd.read_csv("../Dataset/features.csv")
label = pd.read_csv("../Dataset/labels.csv")

sex_column = features["SEX"]
X = features.drop(columns="SEX")
y = label

X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sex_column, test_size=0.2, random_state=0
)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
sex_test = sex_test.reset_index(drop=True)

models = {
    "RandomForest": joblib.load("../joblibs/RandomForest_BestModel_08159.joblib"),
    "GradientBoost": joblib.load("../joblibs/GradientBoost_BestModel_08158.joblib"),
    "AdaBoost": joblib.load("../joblibs/AdaBoost_BestModel_08038.joblib"),
    "Stacking": joblib.load("../joblibs/Stacking_BestModel_08194.joblib"),
}

True_positive = {}
True_negative = {}
False_positive = {}
False_negative = {}

diff_valeur = [1.0, 2.0]

for name, model in models.items():
    for i in diff_valeur:
        y_train_flattened = np.ravel(y_train)
        model.fit(X_train, y_train_flattened)

        y_pred = model.predict(X_test)

        mask = (sex_test == i)

        conf_matrix = confusion_matrix(y_test[mask], y_pred[mask])
        tn, fp, fn, tp = conf_matrix.ravel()

        key = f"{i}_{name}"
        True_positive[key] = tp
        True_negative[key] = tn
        False_positive[key] = fp
        False_negative[key] = fn

        print(f"Confusion matrix for SEX={i} using {name}:")
        print(conf_matrix)
        print("\n" + "=" * 50 + "\n")

for key in True_positive.keys():
    print(f"Modèle et sexe : {key}")
    print(f"  TP: {True_positive[key]}")
    print(f"  TN: {True_negative[key]}")
    print(f"  FP: {False_positive[key]}")
    print(f"  FN: {False_negative[key]}\n")

Taux_TP = {}
Taux_TN = {}
for k in True_positive.keys():
    Taux_TP[k] = True_positive[k]/(True_positive[k]+False_negative[k])
    Taux_TN[k] = True_negative[k]/(True_negative[k]+False_positive[k])


print("\n")

print("Sensibilité : ")
print(Taux_TP)
print("Spécificité : ")
print(Taux_TN)

results_df = pd.DataFrame({
    'Model_Feature': Taux_TP.keys(),
    'True Positive Rate (TPR)': Taux_TP.values(),
    'True Negative Rate (TNR)': Taux_TN.values()
})
print("\nRésumé des métriques :")
print(results_df)