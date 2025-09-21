def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.linear_model import LogisticRegression

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/cancer.csv"

dataframe = pd.read_csv(URL)

target = dataframe["diagnosis"]

features = dataframe[["radius_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "symmetry_mean"]]

model = LogisticRegression()

model.fit(features, target)

print(model.predict([[13.45, 86.6, 555.1, 0.1022, 0.08165, 0.03974, 0.1638]]))