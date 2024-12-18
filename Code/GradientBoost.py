import pandas as pd
import joblib
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

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
gb_model = GradientBoostingClassifier()
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    gb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(scaled_data_train, y_train.values.ravel())
print(f"Meilleurs paramètres pour GradientBoost : {grid_search.best_params_}")
print(f"Meilleure score pour GradientBoost : {grid_search.best_score_}")
y_pred_train = cross_val_predict(gb_model, scaled_df_train, y_train.values.ravel())
gb_model.fit(scaled_df_train, y_train.values.ravel())
y_pred_test = gb_model.predict(scaled_df_test)

metrics = {
    "accuracy_train": accuracy_score(y_train, y_pred_train),
    "accuracy_test": accuracy_score(y_test, y_pred_test),
    "classification_report_train": classification_report(y_train, y_pred_train),
    "classification_report_test": classification_report(y_test, y_pred_test),
    "confusion_matrix_train": confusion_matrix(y_train, y_pred_train),
    "confusion_matrix_test": confusion_matrix(y_test, y_pred_test),
}
print("GradientBoost")
print("-------------------------")
print(metrics)
print("-------------------------")
results = []
results["GradientBoost"] = metrics

print(f"Accuracy (train): {metrics['accuracy_train']}")
print(f"Accuracy (test): {metrics['accuracy_test']}")
print("\nClassification Report (test):\n", metrics['classification_report_test'])
print("\nConfusion Matrix (test):\n", metrics['confusion_matrix_test'])

accuracy = str(metrics['accuracy_test']).split('.')[1][:4]
joblib.dump(gb_model, "GradientBoost_0{accuracy}.joblib")
