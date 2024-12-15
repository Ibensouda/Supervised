from folktables import ACSDataSource, ACSIncome

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSIncome

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
ca_data = data_source.get_data(states=["CA"], download=True)

ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)

ca_features.to_csv('ca_features.csv', index=False)
ca_labels.to_csv('ca_labels.csv', index=False)

pd.DataFrame(ca_labels, columns=['label']).to_csv('ca_labels.csv', index=False)


df_features = pd.read_csv('ca_features.csv')

# hist = df_features.hist(bins=3)
# plt.show()