import numpy as np
import joblib
import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


features = pd.read_csv("ca_features.csv")
label = pd.read_csv("ca_labels.csv")

X_train, X_test, y_train, y_test= train_test_split(
    features, label, test_size=0.2, random_state=0)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train)


scaled_df = pd.DataFrame(scaled_data, columns=X_train.columns)

scaled_df.to_csv('scaled_features.csv', index=False)

joblib.dump(scaler, 'scaler.joblib')
