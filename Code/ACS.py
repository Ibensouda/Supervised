import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


features = pd.read_csv('../Dataset/features.csv', index=False)
labels = pd.read_csv('../Dataset/labels.csv', index=False)

pd.DataFrame(labels, columns=['label']).to_csv('../Dataset/labels.csv', index=False)

df_features = pd.read_csv('../Dataset/features.csv')

hist = df_features.hist(bins=3)
plt.show()